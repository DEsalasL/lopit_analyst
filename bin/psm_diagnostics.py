import os
import gc
import ast
import sys
import time
import pathlib
import lopit_utils
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = 'Dayana E. Salas-Leiva'
__version__ = '0.0.1'
__email__ = 'dayanasalas@gmail.com'


#   ---  Definitions - end  ---  #
def create_dir(dirname, outname):
    cwd = os.getcwd()
    my_dir = os.path.join(cwd, f'{dirname}_{outname}')
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)
    else:
        print('Directory exists')
    return my_dir


def read_matrix(filein, outname, out, copy_cols):
    # dtype for each column is automatically recognized
    df = pd.read_csv(filein, sep='\t', header=0, engine='python',
                     na_values=['NA', '', 0])
    if out:
        df.to_csv(f'{outname}.tsv', sep='\t', index=False,
                  na_rep='NA')
    all_columns = df.columns.to_list()
    if 'PSMs Peptide ID' in all_columns:
        del df['PSMs Peptide ID']
    if 'PSMs.Peptide.ID' not in all_columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'PSMs.Peptide.ID'}, inplace=True)
    df['PSMs.Peptide.ID'] = df['PSMs.Peptide.ID'].astype('object')
    df.columns = df.columns.str.replace('[ |-]', '.', regex=True)
    df.columns = df.columns.str.replace('Abundance.', 'TMT', regex=False)
    df['Experiment'] = df['Spectrum.File'].str.extract(r'(PL\w+)_')
    df['Experiment'] = df['Experiment'].str.replace('_', '', regex=False)
    return df


def p9hist(odf, col1, col2, outname, binwidth, intercept,
           density=False, transform=False, adjust=False):
    # slicing df
    df = odf.loc[:, [col1, col2]]
    # defining object
    if density:
        obj = p9.ggplot(df, p9.aes(col1, colour=col2, fill=col2,
                                   group=col2))
        # density histogram parameters
        obj += p9.geom_density(alpha=0.1, show_legend=False, na_rm=True)
    else:
        obj = p9.ggplot(df, p9.aes(col1, p9.after_stat('density'),
                                   colour=col2, group=col2))
    # if scale need to be adjusted
    if transform:
        obj += p9.scale_x_log10()
    # histogram parameters
    # Note: bin width variations will render object with after_stat('density')
    # as a graph similar to density but without large memory requests.
    obj += p9.geom_histogram(alpha=0.1, binwidth=binwidth,
                             show_legend=False, na_rm=True)
    # general
    obj += p9.facet_grid(f'~{col2}') + p9.theme_bw()
    # adjust graph aspect
    if adjust:
        obj += p9.theme(aspect_ratio=1, panel_grid=p9.element_blank(),
                        axis_text_x=p9.element_text(size=7, hjust=1, vjust=1,
                                                    angle=45),
                        plot_title=p9.element_text(size=14, face='bold'))
    else:
        obj += p9.theme(aspect_ratio=1, panel_grid=p9.element_blank(),
                        axis_text_x=p9.element_text(size=7, hjust=1, vjust=1),
                        plot_title=p9.element_text(size=14, face='bold'))
    # intercept requested
    if intercept != '':
        obj += p9.geom_vline(xintercept=intercept, linetype='dotted',
                             colour='red')
    obj += p9.labels.ggtitle(outname)
    # saving to pdf with custom page size
    custom_width, custom_height = lopit_utils.pdf_size(width=1, height=0.25)
    outpath = os.path.join(os.getcwd(), f'{outname.replace(' ', '_')}.pdf')
    obj.save(outpath, dpi=300, width=custom_width, height=custom_height)
    return outpath


def scatter_plot(odf, col1, col2, outname):
    # slicing df
    df = odf.loc[:, [col1, col2, 'Experiment']]
    # defining object
    obj = p9.ggplot(df, p9.aes(col1, col2)) + p9.geom_point(size=0.5)
    obj += p9.geom_abline(intercept=[-5, 0, 5], slope=0, linetype='dotted',
                          colour='red')
    obj += p9.labels.ggtitle(outname)
    obj += p9.facet_wrap(facets='Experiment')
    # saving to pdf with custom page size
    outpath = os.path.join(os.getcwd(), f'{outname}.pdf')
    custom_width, custom_height = lopit_utils.pdf_size(width=1, height=0.5)
    obj.save(outpath, dpi=300, width=custom_width, height=custom_height)
    return outpath


def large_pivot_tab(odf, d, outname, out, psms=True, reconstituted=False):
    df = odf.copy(deep=True)
    # slice and convert to long
    tag_names = {i: d[i][0] for i in d.keys()}
    df.rename(columns=tag_names, inplace=True)
    if psms:
        sel_cols = list(df.filter(regex=r'^TMT.*\d+').columns) + \
                    ['PSMs.Peptide.ID', 'Experiment']
        sliced_df = df.loc[:, sel_cols]
        ldf = pd.wide_to_long(sliced_df, stubnames='TMT', j='Tag',
                              i=['PSMs.Peptide.ID'])
    else:
        sel_cols = list(df.filter(regex=r'^TMT.*\d+').columns) + \
                   ['Accession', 'Experiment']
        sliced_df = df.loc[:, sel_cols]
        ldf = pd.wide_to_long(sliced_df, stubnames='TMT', j='Tag',
                              i=['Accession', 'Experiment'])
    ldf.reset_index(inplace=True)
    inverted_dic = {int(tag_names[k].split('T')[-1]): k
                    for k in tag_names.keys()}
    ldf['Tag'] = ldf['Tag'].map(inverted_dic)
    ldf['Tag'].str.replace('TMT126N', 'TMT126')
    ldf.rename(columns={'TMT': 'Abundance'}, inplace=True)
    outfile = os.path.abspath(os.path.join(os.getcwd(), f'{outname}.tsv'))
    ldf.to_csv(outfile, sep='\t', index=False, na_rep='NA')
    _ = gc.collect()
    pivot_df = pd.read_csv(outfile, sep='\t', header=0)
    while ldf.shape != pivot_df.shape:
        print('Waiting for ldf to be re-loaded before removing source file')
        time.sleep(0.1)
    pathlib.Path(outfile).unlink(missing_ok=True)
    return pivot_df


def slice_treatment(df, regex, exp, d):
    tmt_cols = df.filter(regex=regex).columns.to_list()
    try:
        c1 = tmt_cols + ['Accession', f'Experiment.{exp}']
        df_1 = df.loc[:, c1]
    except:
        c1 = tmt_cols + ['Accession', 'Experiment']
        df_1 = df.loc[:, c1]
    c2 = [c.replace(f'.{exp}', '') for c in c1]
    col_names = dict(zip(c1, c2))
    df_1.rename(columns=col_names, inplace=True)
    if d == '':
        d = lopit_utils.set_tags(lopit_utils.combos,
                                 lopit_utils.colors, tmt_cols)
    slice = large_pivot_tab(df_1, d, f'slice_{exp}_tmp', False,
                            False)
    return slice, d


def irs_pivot_table(df, taginf):
    # old regex = r'^TMT.*1$' and r'^TMT.*1$'
    slice1, d1 = slice_treatment(df, r'^TMT.*1$', 1, taginf)
    slice2, d2 = slice_treatment(df, r'^TMT.*2$', 2, taginf)
    # slice.sort_values(['Experiment', 'Accession','Tag'],
    #            ascending=[True, True,True], inplace=True)
    return slice1, slice2, d1, d2


def p9box(odf, tag_col, outname):
    df = odf.copy(deep=True)
    tag_colors = {i: tag_col[i][1] for i in tag_col.keys()}
    obj = p9.ggplot(df, p9.aes(x='Tag', y='Abundance', fill='Tag'))
    obj += p9.geom_boxplot(outlier_alpha=0.1, outlier_size=0.5,
                           show_legend=False, na_rm=True)
    obj += p9.scale_fill_manual(values=tag_colors)
    obj += p9.scale_y_log10()
    obj += p9.facet_grid('~Experiment')
    obj += p9.theme_bw()
    obj += p9.theme(aspect_ratio=3,
                    axis_text_x=(p9.element_text(angle=50, hjust=1,
                                                 vjust=1, size=5)),
                    axis_title=p9.element_text(size=8, face='bold'),
                    panel_grid_major_x=p9.element_blank(),
                    panel_grid_minor_y=p9.element_blank(),
                    plot_title=p9.element_text(size=14, face='bold'))
    obj += p9.labels.ggtitle(outname)
    # save pdf to custom size pdf
    custom_width, custom_height = lopit_utils.pdf_size(width=1, height=0.5)
    outpath = os.path.join(os.getcwd(), f'{outname}.Boxplot.pdf')
    obj.save(outpath, dpi=300, width=custom_width, height=custom_height)
    return outpath


def catplot(new_df, tag_col, outname):
    tag_colors = {i: tag_col[i][1] for i in tag_col.keys()}
    plt.figure(figsize=(15, 10))
    sns.set_theme()
    wrap_at = int(np.ceil(len(set(new_df.Experiment.to_list()))/2))
    g = sns.catplot(data=new_df, x='Tag', y='Abundance', col='Experiment',
                    col_wrap=wrap_at, kind='box', palette=tag_colors,
                    errorbar=None)
    g.set(yscale="log")
    for ax in g.axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_size(8)
    g.fig.suptitle(outname, y=1)
    file_out = os.path.join(os.getcwd(), f'{outname}.box.plot.png')
    plt.savefig(file_out)
    return file_out


def comparative_box_plots(list_w_dfs_tup, taginf, writeout):
    loaded_plots = []
    for entry in list_w_dfs_tup:
        df, prefix = entry
        pivot_df = large_pivot_tab(odf=df,
                                   d=taginf,
                                   outname=f'large_pivot_df_{prefix}',
                                   out=writeout,
                                   psms=True,
                                   reconstituted=False)
        boxplot = p9box(odf=pivot_df,
                        tag_col=taginf,
                        outname='Tag Abundance by experiment - '
                        f'{prefix}-filtering')
        loaded_plots.append(boxplot)
    return loaded_plots


def comparative_hist(list_w_dfs_tup, density, taginf):
    loaded_plots = []
    for entry in list_w_dfs_tup:
        df, prefix = entry
        tottmt_df = tmt_by_exp(df, taginf, 'TMT.Abundance.by.PSM')
        hist = p9hist(tottmt_df, 'TMT.Abundance.by.PSM', 'Experiment',
                      f'TMT Abundance by Exp - {prefix}-filtering',
                      0.1, '', density, True,
                      True)
        loaded_plots.append(hist)
    return loaded_plots


def trend_plot(df, col1, col2, col3, tag_cols):
    tag_colors = {i: tag_cols[i][1] for i in tag_cols.keys()}
    sdf = df.loc[:, [col1, col2, col3, 'Experiment']]
    obj = p9.ggplot(sdf, p9.aes(x=col1, y=col2, fill=col3, group='Experiment'))
    # add trendline
    obj += p9.geom_smooth(method='lm')
    obj += p9.geom_point()
    obj += p9.scale_y_log10()
    obj += p9.scale_fill_manual(values=tag_colors)
    obj += p9.facet_wrap(facets='Experiment', nrow=3)
    obj += p9.labs(x='Peptide amount', y='Total abundance')
    obj += p9.theme_bw()
    obj += p9.theme(panel_grid_major_x=p9.element_blank(),
                    panel_grid_minor_y=p9.element_blank())
    # print(obj)
    return obj


def trend(pvt, phenodata, tag_col, missing_cols):
    # remove missing channels due to plex differences
    channels = list(set([val for entry in missing_cols.values()
                         for val in entry]))
    index_tor = list(pvt[(pvt.Experiment.isin(list(missing_cols.keys()))) &
                         (pvt.Abundance.isnull()) &
                         (pvt.Tag.isin(channels))].index)
    # df without missing channels by experiment
    mod_pvt = pvt[~pvt.index.isin(index_tor)]
    del index_tor
    gc.collect()
    grouped = mod_pvt.groupby(['Tag', 'Experiment']).sum()
    grouped.rename(columns={'Abundance': 'Total.Abundance'}, inplace=True)
    grouped.reset_index(inplace=True)
    merged = pd.merge(phenodata, grouped, on=['Tag', 'Experiment'],
                      how='inner')
    # faceted trend plot in a single file
    global_smooth = trend_plot(merged, 'Peptide.amount',
                               'Total.Abundance', 'Tag', tag_col)

    # saving to pdf with custom page size
    custom_width, custom_height = lopit_utils.pdf_size(width=1, height=0.5)
    outpath = os.path.join(os.getcwd(), f'P.All.Exp_PeptideAbund.trend.pdf')
    global_smooth.save(outpath, dpi=300, width=custom_width,
                       height=custom_height)
    return outpath


def tmt_by_exp(df, d, colname):
    # summation of TMTs by entry and Experiment
    tots = pd.DataFrame(df.groupby(['Experiment', 'PSMs.Peptide.ID']
                                   )[list(d)].agg(sum).sum(axis=1))
    tots.rename(columns={0: colname}, inplace=True)
    tots.reset_index(inplace=True)
    return tots


def missing_cols_by_experiment(df):
    df_copy = df.copy(deep=True)
    all_cols = list(df_copy.filter(regex=r'^TMT.*\d+').columns)
    tor = {}
    for exp, sdf in df_copy.groupby('Experiment', as_index=False):
        sdf.dropna(axis=1, how='all', inplace=True)
        avail_cols = list(sdf.filter(regex=r'^TMT.*\d+').columns)
        missing_cols = list(set(all_cols).difference(set(avail_cols)))
        if missing_cols:
            tor[exp] = missing_cols
    print('Missing channels by experiment:\n', len(tor), tor)
    return tor


def run_diagnostics(psmfile, phenotypefile, density, writeout,
                    outname):

    print('\n*** - Beginning of diagnostics workflow - ***\n')
    #  ---  Matrix pre-processing step 1
    #  Creating a directory and moving into it  #
    dfs_dir_path = create_dir('Step1_',
                              f'Diagnostics_{outname}')
    print(dfs_dir_path)
    os.chdir(dfs_dir_path)

    #  reading-in workflow parameters
    if isinstance(writeout, str):
        out = ast.literal_eval(writeout)
    pre_parsed_psm = pd.read_csv(psmfile, sep='\t', header=0,
                                 engine='python')
    # patch to eliminate blank spaces in quantitative data (bug in PD3.1)
    tmt_cols_prep = list(pre_parsed_psm.filter(regex=r'^Abundance ').columns)
    for col in tmt_cols_prep:
        pre_parsed_psm[col] = pre_parsed_psm[col].replace(' ', '')
        # pre_parsed_psm[col] = pre_parsed_psm[col].astype('float')

    # customizing taginf for pre_parsed_psm
    taginf = lopit_utils.custom_taginf(lopit_utils.taginf, pre_parsed_psm)
    #
    if phenotypefile is not None:
        pre_parsed_pheno = pd.read_csv(phenotypefile, sep=r'\t|\,', header=0)
    else:
        pre_parsed_pheno = pd.DataFrame()
    if isinstance(density, str):
        density = ast.literal_eval(density)

    #   ---
    script_path = os.path.realpath(__file__)
    with open(f'Command_line.{outname}.txt', 'w') as output:
        line = f'program: {script_path}\ninput: {os.path.abspath(psmfile)}\n'
        line += f'accessory: {os.path.abspath(phenotypefile)}\n'
        line += f'density: {str(density)}\nwriteout: {str(writeout)}'
        output.write(line)
    #   ---
    statistics_df = pd.DataFrame(pre_parsed_psm .describe(include='all'))

    pre_parsed_psm.to_csv(f'Parsed_PSM.headers.{outname}.tsv',
                              sep='\t', index=False)
    # checking phenodata experiment declaration and experiments in PSMs file
    _ = lopit_utils.experiments_exist(pre_parsed_psm, pre_parsed_pheno,
                                      'psms', 'pheno')
    if writeout:
        statistics_df.to_csv(f'Statistics.test.{outname}.tsv',
                             sep='\t', index=True)  # see README1

    #   ---   Calculating TMT labeling efficiency   ---   #
    print('Calculating TMT labeling efficiency...')
    _ = lopit_utils.ecalculator(pre_parsed_psm, outname)

    #  ---  diagnostic density plots  ---  #
    #
    print('Generating plots ...')
    mass_error = p9hist(odf=pre_parsed_psm,
                        col1='Delta.M.in.ppm',
                        col2='Experiment',
                        outname='PSMs delta Masses in ppm',
                        binwidth=0.05,
                        intercept='',
                        density=density,
                        transform=False,
                        adjust=True)

    precursor_i_int = p9hist(odf=pre_parsed_psm,
                             col1='Intensity',
                             col2='Experiment',
                             outname='Intensity',
                             binwidth=0.05,
                             intercept='',
                             density=density,
                             transform=True,
                             adjust=True)


    isol_interf = p9hist(odf=pre_parsed_psm,
                         col1='Isolation.Interference.in.Percent',
                         col2='Experiment',
                         outname='Isolation Interference',
                         binwidth=5,
                         intercept=50,
                         density=density,
                         transform=False,
                         adjust=True)

    sps_mass_matches = p9hist(odf=pre_parsed_psm,
                              col1='SPS.Mass.Matches.in.Percent',
                              col2='Experiment',
                              outname='SPS Mass Matches',
                              binwidth=10,
                              intercept=[45, 65],
                              density=density,
                              transform=False,
                              adjust=True)

    signal_noise_ratio = p9hist(odf=pre_parsed_psm,
                                col1='Average.Reporter.SN',
                                col2='Experiment',
                                outname='Average Reporter SN ratio',
                                binwidth=0.05,
                                intercept=5,
                                density=density,
                                transform=True,
                                adjust=True)

    ion_trap = p9hist(odf=pre_parsed_psm,
                      col1='Ion.Inject.Time.in.ms',
                      col2='Experiment',
                      outname='Ion Injection Time in ms',
                      binwidth=2,
                      intercept='',
                      density=density,
                      transform=False,
                      adjust=True)


    tot_tmt_df = tmt_by_exp(pre_parsed_psm, taginf,
                            colname='TMT.Abundance.by.PSM')
    tot_tmt_plot = p9hist(odf=tot_tmt_df,
                          col1='TMT.Abundance.by.PSM',
                          col2='Experiment',
                          outname='TMT Abundance by Experiment-PSMs',
                          binwidth=0.1,
                          intercept='',
                          density=density,
                          transform=True,
                          adjust=True)

    rt_scatterplot = scatter_plot(odf=pre_parsed_psm,
                                  col1='RT.in.min',
                                  col2='Delta.M.in.ppm',
                                  outname='Retention time')

    #
    # # #  ---  diagnostic box plots  ---  #

    odf_copy = pre_parsed_psm.copy(deep=True)
    pivot_df = large_pivot_tab(odf_copy,
                               d=taginf,
                               outname='large_pivot_df',
                               out=writeout,
                               psms=True,
                               reconstituted=False)
    boxplot = p9box(odf=pivot_df,
                    tag_col=taginf,
                    outname='Tag Abundance by Experiment')

    mis_cols = missing_cols_by_experiment(odf_copy)
    if not pre_parsed_pheno.empty:
        trendplot = trend(pivot_df, pre_parsed_pheno, taginf, mis_cols)

        tmt = p9hist(odf=pivot_df,
                     col1='Abundance',
                     col2='Experiment',
                     outname='TMT Abundance by Experiment - notch artifact',
                     binwidth=0.05,
                     intercept='',
                     density=density,
                     transform=True,
                     adjust=True)

    else:
        tmt = None
        trendplot = None

    # --- rendering histograms
    if tmt is not None:
        all_histograms = [mass_error, precursor_i_int, isol_interf,
                          sps_mass_matches, signal_noise_ratio, ion_trap,
                          tot_tmt_plot, rt_scatterplot, trendplot, tmt, boxplot]
    else:
        all_histograms = [mass_error, precursor_i_int, isol_interf,
                          sps_mass_matches, signal_noise_ratio, ion_trap,
                          tot_tmt_plot, rt_scatterplot, boxplot]

    # create a single pdf from multiple pdfs
    fileoutpath = os.path.join(os.getcwd(), f'All_diagnostic_hist.pdf')
    _ = lopit_utils.merge_pdfs(all_histograms, outname=fileoutpath)

    os.chdir('../..')
    print('\n*** - End of diagnostics workflow - ***\n')
    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')
    return pre_parsed_psm, pre_parsed_pheno


# --- program ---  #


if __name__ == '__main__':
    '''
    script.py <PSM file> <phenotype data> <density: True or False> <write files>
    PSM file: 
    dir= D:\PycharmProjects\LOPIT\Perkinsus_LOPIT_K\Data\
    dir += \\Nov_2021_Mascot\kb601_20211127_PmarLOPIT_Mascot_multicon_PSMs.tsv 
    Experiment data: dir += PL_pData.csv
    density False
    write files True
    rename_cols = {TMT131:TMT131N, TMT128N:TMT128}
    '''
    psm = sys.argv[1]
    pheno_Data = sys.argv[2]
    density = sys.argv[3]
    write = sys.argv[4]
    outname = sys.argv[5]
    _ = run_diagnostics(psm, pheno_Data, density, write, outname)
    print('Program has finished')
