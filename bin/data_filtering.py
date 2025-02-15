import os
import gc
import sys
import ast
import numpy as np
import pandas as pd
import lopit_utils
import psm_diagnostics as dia


def safe_gard():
    dic = {'Quan.Info': '', 'Marked.as': '',
           'Number.of.Protein.Groups': 100, 'Rank': 100,
           'Search.Engine.Rank': 100, 'Concatenated.Rank': 100,
           'Isolation.Interference.in.Percent': 100,
           'SPS.Mass.Matches.in.Percent': 0, 'Average.Reporter.SN': 0}
    return dic


def df_reduction(odf, exclude, sn_value):
    df = odf.copy(deep=True)
    original = summary_stats(df, 'Master.Protein.Accessions', 'Pre',
                             'original')
    # fillna with values that ensure data elimination based on conditions below
    df.fillna(value=safe_gard(), inplace=True)
    # drop unidentified accessions
    df.dropna(subset=['Master.Protein.Accessions'], inplace=True)
    nonnan = summary_stats(df, 'Master.Protein.Accessions', 'Post',
                           'with Master Protein identification')

    #   ---  conditions for data to keep  ---   #
    k = (
        (df['PSM.Ambiguity'] == 'Unambiguous') &  # retain only unambiguous PSMs
        (df['Contaminant'] == False) &  # PSMs annotated as contaminants
        ~(df['Master.Protein.Accessions'].str.contains('Cont_')) &  # contaminant
        ~(df['Protein.Accessions'].str.contains('Cont_')) &  # contaminants
        ~(df['Marked.as'].str.contains(';')) &  # ambiguous
        ~(df['Marked.as'].str.contains(exclude)) &  # contaminants, see bypass
        (df['Number.of.Protein.Groups'] == 1) &  # match a unique protein group
        (df['Rank'] == 1) &  # PSMs of rank = 1
        (df['Search.Engine.Rank'] == 1) &  # PSMs of rank = 1 by search engine
        (df['Concatenated.Rank'] == 1) &  # target and decoy PSMs matched to a
        # given spectrum and ranked by their score
        (df['MS.Order'] == 'MS2') &  # PSMs identified for MS2-spectra
        (df['Isolation.Interference.in.Percent'] <= 50) &  # remove PSMs with
        # isolation interference > 50 %
        (df['SPS.Mass.Matches.in.Percent'] >= 50) &  # remove PSMs with fewer
        # than half SPS precursors matched
        (df['Ion.Inject.Time.in.ms'] <= 50) &  # keep PSMs with inject time < 50
        (df['Average.Reporter.SN'] >= sn_value) &  # remove PSMs with low S/N
        (df['Quan.Info'] != 'NoQuanLabels')  # remove entries == "NoQuanLabels"
        )

    new_df = df.loc[k]
    pre = summary_stats(df, 'Master.Protein.Accessions', 'Pre',
                        '(NoQuanLabels)')
    post = summary_stats(new_df, 'Master.Protein.Accessions', 'Post',
                         '(NoQuanLabels)')
    cat_stats = pd.concat([original, nonnan, pre, post])
    cat_stats.drop_duplicates(inplace=True)
    cat_stats.to_csv('Filtering_1-report.tsv', sep='\t', index=True,
                     na_rep='NA')
    if new_df.empty:
        print(f'FATAL ERROR caused by an empty dataframe.\n'
              f'This means that first filtering step failed because of empty '
              f'or nan values in at least one of the following columns:\n'
              f'{safe_gard().keys()}')
        sys.exit(-1)
    return new_df


def summary_stats(df, col, bname, m):
    grp = df.groupby('Experiment')
    # f = pd.DataFrame(grp[col].apply(lambda x: len(x)))
    u = pd.DataFrame(grp[col].apply(lambda x: len(set(x))))
    u['Status/Dataset type'] = f'{bname} unique {m}'
    print(f'***  {bname}-filtering {m}: '
          f'total UNIQUE proteins in dataset is: \n{u}***')
    return u


def remove_cols_mval(df, threshold):
    ls = []
    for col in df.columns.to_list():
        size = len(df[col])
        t = [type(i) for i in df[col].to_list()]
        print('df[col', df[col].to_list())
        print(t)
        nans = len([i for i in df[col] if np.isnan(i)])
        if nans / size > threshold:
            ls.append(col)
    if ls:
        df.drop(ls, inplace=True, axis=1)
        print(f'The following columns contain over {threshold} MV and have'
              f' been removed:\n{ls}')
    return df


def missing_values_calculation(df, verbosity):
    dfs = []
    for experiment, subdf in df.groupby('Experiment'):
        subdf.dropna(axis=1, how='all', inplace=True)
        sdf_cols = list(subdf.filter(regex=r'^TMT.*\d+').columns)
        s = lopit_utils.nan(subdf, sdf_cols)
        dfs.append(s)

    # writing first level filtered dfs by experiment
    if verbosity:
        for subdf in dfs:
            exp = list(set(subdf['Experiment'].to_list()))
            subdf.to_csv(f'mv_calculated_by_{exp[0]}.tsv', sep='\t',
                         index=False, na_rep='NA')

    #   ---   concatenate first level filted subdfs and writing   ---
    full_df = pd.concat(dfs)
    full_df.to_csv(f'mv_calculated_full_df.tsv', sep='\t',
                   index=False, na_rep='NA')
    return full_df


def does_exclude_exists(df, exclude):
    if exclude == ';':  # force bypass
        return True
    my_series = list(set(df['Marked.as'].str.contains(exclude).to_list()))
    if True in my_series:
        return True
    else:
        print('Error: taxon name does not exist in PSMs file')
        sys.exit(-1)


def run_data_filter(arg1, density, fileout, outname, exclude, sn_value):

    print('\n*** - Beginning of data filtering workflow - ***\n')
    directory = lopit_utils.create_dir('Step2_',
                                       f'First_filter_{outname}')
    os.chdir(directory)

    #   ---  Program  ---   #
    if isinstance(fileout, str):
        fileout = ast.literal_eval(fileout)
    if isinstance(density, str):
        density = ast.literal_eval(density)
    if isinstance(arg1, pd.DataFrame):
        pre_parsed_psm = arg1
    else:
        pre_parsed_psm = pd.read_csv(arg1, sep='\t', header=0,
                                     engine='python')

    # check if declared taxon name exists in 'Marked.as' field.
    if exclude is None or exclude.lower() == 'other':  # bypass when
        # no other taxon is expected in pd out
        exclude = ';'
    _ = does_exclude_exists(pre_parsed_psm, exclude)

    # proceed with initial reduction
    filt_df = df_reduction(pre_parsed_psm, exclude, sn_value)
    taginf = lopit_utils.custom_taginf(lopit_utils.taginf, pre_parsed_psm)

    #   ---
    script_path = os.path.realpath(__file__)
    with open(f'Command_line.txt', 'w') as output:
        line = f'program: {script_path}\ninput: {os.path.abspath(arg1)}\n'
        line += f'density: {str(density)}\nwriteout: {str(fileout)}'
        output.write(line)
    if fileout:
        filt_df.to_csv('Filtered1_df.tsv', sep='\t', index=False)

    dfs_to_compare = [(pre_parsed_psm.copy(deep=True), 'pre'),
                      (filt_df.copy(deep=True), 'post')]
    boxplots = dia.comparative_box_plots(dfs_to_compare, taginf, fileout)
    histoplots = dia.comparative_hist(dfs_to_compare, density, taginf)
    comp_plots = boxplots + histoplots
    my_plots = lopit_utils.pw_object_layout(comp_plots)[len(comp_plots)]
    print('Rendering plots ...')
    _ = lopit_utils.rendering_figures(my_plots,
                                      'Comparative_plots.filter1.pdf')
    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')
    processed_df = missing_values_calculation(filt_df, fileout)
    os.chdir('../..')
    print('*** - Filtering  workflow has finished -***\n')
    return processed_df


#  ---  Execute  ---  #
'''
script.py <original PSM from PD> <density: True or False>
PSM = D:\PycharmProjects\LOPIT\Perkinsus_LOPIT_K\Data\
PSM += \\Nov_2021_Mascot\kb601_20211127_PmarLOPIT_Mascot_multicon_PSMs.txt 
density: False
write file out: False
outname: name to write out
'''
if __name__ == '__main__':
    # _ = run_data_filter(sys.argv[1], sys.argv[2], sys.argv[3],
    #                     sys.argv[4], sys.argv[5])
    print('Program has finished')
