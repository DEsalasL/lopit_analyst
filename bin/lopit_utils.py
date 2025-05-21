import csv
import os
import re
import gc
import sys
import json
import string
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from functools import reduce
import plotly.express as px
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt
from pypdf import PdfWriter, PdfReader
# from SVM_KNN_RF_clustering import write_mydf
from itertools import combinations, zip_longest
from matplotlib.backends.backend_pdf import PdfPages
from dash import Dash, dcc, html, Input, Output, callback

#   ---   available TMT channels, internal equivalences, and colors ---   #

#   ---   colors and tmt channels   ---   #
tmt_chans = ['TMT126', 'TMT127N', 'TMT127C', 'TMT128N', 'TMT128C',
             'TMT129N', 'TMT129C', 'TMT130N', 'TMT130C', 'TMT131N',
             'TMT131C', 'TMT132N', 'TMT132C', 'TMT133N', 'TMT133C',
             'TMT134N', 'TMT134C', 'TMT135N']

colors = ['gold', '#fb9a99', '#e31a1c', '#a6cee3', '#1f78b4',
          '#fdbf6f', '#ff7f00', '#e3e3e3', '#6a3d9a', '#b2df8a',
          '#33a02c', '#f720e5', '#66071c', '#0fa6f7', '#4d4343',
          '#91e2e3', '#013738', '#f57ad6', '#75391e', '#827542',
          '#718063', '#03558c', '#f1f2a0', '#1c038c', '#3e14fa',
          '#ffe0f1', '#7502f7', '#f2555d', '#0aedf5', '#e6f7e1',
          '#a9c1f1', '#5c3641', '#f7bdf3', '#98c2a7', '#999b84',
          '#f6546a', '#bada55', '#ffc0a1', '#a8abe0', '#02a4d3',
          '#2c2422', '#f4e9e5', '#552e65', '#4a5722', '#f8fbee',
          '#89856f', '#f9f8f5', '#0d1611', '#e6f8ee', '#03182c']



#   --- patch for bug in large pivot table function  ---   #

#   ---



def name_combos():
    digits = list(string.digits[1:])
    combos = []
    for comb in combinations(digits, 4):
        c = ''.join(comb)
        combos.append(f'TMT{c}')
    return combos


def set_tag_info(long_lst, short_lst):
    l = []
    for i in (list(p for p in pair) for pair in zip_longest(long_lst, short_lst)
              if None not in pair):
        l.append(i)
    return l


combos = name_combos()


def set_tags(names, colors, tmt_chans):
    c = set_tag_info(names, colors)
    return {entry[1]: entry[0] for entry in set_tag_info(c, tmt_chans)}


taginf = set_tags(combos, colors, tmt_chans)


def custom_taginf(tags, df):
    all_cols = list(df.filter(regex=r'^TMT.*\d+').columns)
    return {k: tags[k] for k in tags.keys() if k in all_cols}


#   --- end of patch for large pivot table module bug  ---   #

#    ---  writing command lines ---   ###
def command_line_out(suffix, **kwargs):
    line = ''
    for k in kwargs.keys():
        if k == 'subparser_name':
            line += f'{k}: {os.path.realpath(__file__)}\n'
        else:
            line += f'{k}: {kwargs[k]}\n'
    with open(f'command_line_out_{suffix}.txt', 'w') as out:
        out.write(line)
    return 'Done'


#  --- patchworklib object layout


def pw_object_layout(lst_w_pw_obj):
    if len(lst_w_pw_obj) == 1:
        return {1: (lst_w_pw_obj[0])}
    elif len(lst_w_pw_obj) == 2:
        return {2: (lst_w_pw_obj[0] | lst_w_pw_obj[1])}
    elif len(lst_w_pw_obj) == 3:
        return {3: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2])}
    elif len(lst_w_pw_obj) == 4:
        return {4: (lst_w_pw_obj[0] | lst_w_pw_obj[1]) |
                   (lst_w_pw_obj[2] | lst_w_pw_obj[3])}
    elif len(lst_w_pw_obj) == 5:
        return {5: (lst_w_pw_obj[0] | lst_w_pw_obj[1]) |
                   (lst_w_pw_obj[2] | lst_w_pw_obj[3]) / (lst_w_pw_obj[4])}
    elif len(lst_w_pw_obj) == 6:
        return {6: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2]) /
                   (lst_w_pw_obj[3] | lst_w_pw_obj[4] | lst_w_pw_obj[5])}
    elif len(lst_w_pw_obj) == 7:
        return {7: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2] |
                    lst_w_pw_obj[3]) / (lst_w_pw_obj[4] | lst_w_pw_obj[5] |
                                        lst_w_pw_obj[6])}
    elif len(lst_w_pw_obj) == 8:
        return {8: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2] |
                    lst_w_pw_obj[3]) / (lst_w_pw_obj[4] | lst_w_pw_obj[5] |
                                        lst_w_pw_obj[6] | lst_w_pw_obj[7])}
    elif len(lst_w_pw_obj) == 9:
        return {9: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2]) /
                   (lst_w_pw_obj[3] | lst_w_pw_obj[4] | lst_w_pw_obj[5]) /
                   (lst_w_pw_obj[6] | lst_w_pw_obj[7] | lst_w_pw_obj[8])}
    elif len(lst_w_pw_obj) == 10:
        return {10: (lst_w_pw_obj[0] | lst_w_pw_obj[1] | lst_w_pw_obj[2] |
                     lst_w_pw_obj[3]) /
                    (lst_w_pw_obj[4] | lst_w_pw_obj[5] | lst_w_pw_obj[6] |
                     lst_w_pw_obj[7]) /
                    (lst_w_pw_obj[8] | lst_w_pw_obj[9])}
    else:
        return None


def rendering_figures(dic, out_file):
    if dic is not None:
        dic.savefig(out_file)
        return 'Done'
    else:
        print('Too many objects to draw in a single figure')
        sys.exit(-1)


def compare_loop(list_with_tuples):
    file_paths = []
    for tup in list_with_tuples:
        fig_lst, oname = tup
        if len(fig_lst) > 1:
            my_layout = pw_object_layout(fig_lst)[len(fig_lst)]
        elif len(fig_lst) == 1:
            my_layout = fig_lst[0]
        else:
            print('***  Warning  *** : emtpy figure:\n', fig_lst, oname)
            my_layout = None
        out_name = os.path.join(os.getcwd(), f'{oname}.pdf')
        if my_layout is not None:
            _ = rendering_figures(my_layout, out_name)
            file_paths.append(out_name)
    return file_paths


def multiple_fig_to_pdf(lst, file_out):
    with PdfPages(f'{file_out}.pdf') as pdf_pages:
        for i, fig in enumerate(lst):
            fig = plt.figure(i)
            pdf_pages.savefig(fig)
    return 'Done'


def merge_pdfs(lst_w_pdfs, outname):
    print('Merging PDFs...')
    writer = PdfWriter()
    for path in lst_w_pdfs:
        print(f'Processing PDF file {path}...')
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            raise Exception(f'Error processing PDF file {path}: {e}')

    try:
        with open(outname, 'wb') as output_file:
            writer.write(output_file)
        print(f'PDFs merged successfully into {outname}.')
    except Exception as e:
        raise Exception(f'Error writing merged PDF: {e}')

    # delete source files
    print('Deleting source PDF files...')
    for abspath in lst_w_pdfs:
        fpath = pathlib.Path(abspath)
        fpath.unlink()
    return 'Done'


def merge_images(images_lst, outname):
    images = [Image.open(x) for x in images_lst]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_image.save(f'{outname}.jpg')
    for abspath in images_lst:
        fpath = pathlib.Path(abspath)
        fpath.unlink()
    return 'Done'


def pdf_size(width, height):
    # Calculate plot dimensions for 1/4 of letter size page (adjust as needed)
    pdf_width_inches = 8.5
    pdf_height_inches = 11
    plot_width_inches = pdf_width_inches * width
    plot_height_inches = pdf_height_inches * height
    return plot_width_inches, plot_height_inches


def legend_box_location():
    legend = plt.legend(loc='upper left',
               fontsize=4, bbox_to_anchor=(1.02, 1),
               borderaxespad=0.0)
    plt.gca().add_artist(legend)
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=(0, 0, 0.75, 1))  # [left, bottom, right, top]

#   ---
def create_dir(dirname, outname):
    cwd = os.getcwd()
    dir = os.path.join(cwd, f'{dirname}_{outname}')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    else:
        print('Directory exists')
    return dir


def garbage_collector(list_of_dfs_to_keep):
    alldfs = [var for var in dir() if
              isinstance(eval(var), pd.core.frame.DataFrame)]
    for mem_df in alldfs:
        if mem_df not in list_of_dfs_to_keep:
            del mem_df
    gc.collect()
    return list_of_dfs_to_keep


#  --- Start: matrices correction workflows ---

def experiment_assigment(df, exp_pref, col):
    if col is None:
        print('No column for inferring experiment names has been declared')
        sys.exit(-1)
    groups = []
    rawnames = []
    if isinstance(exp_pref, list) and len(exp_pref) == 2:
        ename, vname = exp_pref[0], exp_pref[1]
        exp_pref = {vname: ename}
    for k in exp_pref.keys():
        raw = list(set(df[col].to_list()))
        rawnames.extend(raw)
        tmp_sdf = df[df[col].str.contains(k)].copy(deep=True)
        if tmp_sdf.empty:
            print(f'Declared {k} for raw file seems incorrect.')
        tmp_sdf['Experiment'] = exp_pref[k]
        groups.append(tmp_sdf)
    new_df = pd.concat(groups)
    if df.shape[0] != new_df.shape[0]:
        print('Available raw file names', '\n'.join(set(rawnames)))
        print(df.shape[0], new_df.shape[0])
        print('There are missing values. Some experiments might not have been'
              'declared correctly.')
        sys.exit(-1)
    return new_df


def renaming_columns(df, rename_cols, outname):
    print('rename columns has been requested as follows:\n', rename_cols)
    tmt_cols = [col for col in df.columns.to_list() if
                col.startswith('TMT')]
    #  --- slicing by requested experiments ---  #
    exps = list(rename_cols.keys())
    #   ---
    sdfs = []
    for exp, sdf in df.groupby('Experiment', as_index=False):
        if exp not in exps:
            nsdf = sdf[~sdf['Experiment'].isin(exps)].copy(deep=True)
            # force to drop if entire column has '' == bug in PD3.1
            for col in tmt_cols:
                nsdf[col].replace('', np.nan, inplace=True)
            nsdf.dropna(axis=1, how='all', inplace=True)
        else:
            nsdf = sdf[sdf['Experiment'].isin(exps)].copy(deep=True)
            # patch for bug created by PD3.1
            target_cols = [rename_cols[exp][k] for k in rename_cols[exp].keys()]
            for target in target_cols:
                empty_list = [e for e in nsdf[target].to_list() if e == '']
                if len(empty_list) == nsdf.shape[0]:
                    print(f'Dropping empty column {target} from '
                          f'experiment {exp}')
                    del nsdf[target]
                else:
                    print('At least one column to rename is not empty and '
                          'will appear redundant with the provided renames\n'
                          'Exiting program')
                    sys.exit(-1)
            # ---
            nsdf.dropna(axis=1, how='all', inplace=True)
            custom_cols = rename_cols[exp]
            nsdf.rename(columns=custom_cols, inplace=True)
        sdfs.append(nsdf)

    cats = pd.concat(sdfs)
    cols = [col for col in df.columns.to_list()
            if col in cats.columns.to_list()]
    cats = cats.reindex(cols, axis=1)
    cats.dropna(axis=1, how='all', inplace=True)

    cats.to_csv(f'{outname}_formatted_PSMs.matrix_w_re-declared_'
                f'channels.tsv', sep='\t', index=False)#, na_rep='NA')
    return cats


def modifications_checkup(df):
    terms = ['N-Term(TMTpro)', 'TMT']
    x= df['Modifications'].str.contains('|'.join(terms))
    modif = [i for i in set(x.tolist()) if i is True][0]
    return modif

def compare_merge(df1, df2):
    merged = pd.merge(df1.loc[:, ['tmp', 'Accession']],
                      df2, left_on='tmp', right_on='Accession_orig',
                      how='outer')
    merged['Accession'] = merged.loc[
        merged['Accession'] == np.nan, 'Accession'] = merged[
        'Accession'].fillna(merged['Accession_orig'])
    merged.drop(['tmp', 'Accession_orig'], axis=1, inplace=True)
    return merged


def accesion_checkup(df1, df2, ftype='marker'):
    df1_accs = list(set(df1['Accession'].to_list()))
    df2_accs = df2['Accession'].to_list()
    intersection = set(df1_accs).intersection(df2_accs)
    missing_acc = [v for v in df1_accs if v not in df2_accs]
    # accessions and not accession-psm:
    if intersection:
        if missing_acc and ftype != 'accessory file':
            print('There following accessions are missing from '
                  f'{ftype}:\n', '\n'.join(missing_acc))
            if ftype == 'marker':
                print('Exiting program...')
                sys.exit(-1)
        return df2
    else:
        # checking if accessions are accession-psm
        df1['tmp'] = df1['Accession'].apply(lambda x: x[0: x.rfind('_')])
        df1_accs = list(set(df1['tmp'].to_list()))
        df2_accs = df2['Accession'].to_list()
        intersection = set(df1_accs).intersection(df2_accs)
        df2.rename(columns={'Accession': 'Accession_orig'}, inplace=True)
        # merge dfs using new names and preserving original source name when nan
        if intersection:
            if ftype == 'marker' and len(intersection) == len(df1_accs):
                merged = compare_merge(df1.loc[:, ['tmp', 'Accession']], df2)
                return merged
            if ftype != 'marker':
                if len(intersection) != len(df1_accs):
                    print(f'There are accessions missing in the {ftype} file')
                merged = compare_merge(df1.loc[:, ['tmp', 'Accession']], df2)
                return merged
            else:
                return None
        else:  # accessions between df1 and df2 does not intersect
            print('There is no intersection between the two compared dfs')
            print('saving compared dfs for debugging.\nExiting program...')
            df1.loc[:, ['tmp', 'Accession']].to_csv(f'df1.partial.debug.tsv',
                                                    sep='\t', index=False)
            df2.to_csv(f'df2.full.{ftype}.debug.tsv', sep='\t', index=False)
            sys.exit(-1)


def psm_matrix_prep(filein, outname, exp_pref, rename_cols, args,
                    usecol, verbose):
    # dtype for each column is automatically recognized
    try:
        df = pd.read_csv(filein, sep='\t', header=0, engine='python',
                         na_values=['NA', ''])
        print(df.columns.to_list())
    except:
        print('file cannot be read as dataframe (it is likely the '
              'wrong file type). Exiting program')
        sys.exit(-1)
    # patch to eliminate blank spaces in quantitative data (bug in PD3.1)
    tmt_cols_prep = df.filter(regex=r'^Abundance ').columns.to_list()
    for col in tmt_cols_prep:
        df[col] = df[col].replace(' ', '')

    #  write command
    _ = command_line_out('psms', **args)

    all_columns = df.columns.to_list()
    if 'PSMs Peptide ID' in all_columns:
        del df['PSMs Peptide ID']
    if 'PSMs.Peptide.ID' not in all_columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'PSMs.Peptide.ID'}, inplace=True)
    df['PSMs.Peptide.ID'] = df['PSMs.Peptide.ID'].astype('object')
    df.columns = df.columns.str.replace('[ |-]', '.', regex=True)
    df.columns = df.columns.str.replace('Abundance.', 'TMT', regex=False)
    #   --- if 0 in TMT replace for np.NA ---   #
    scol = [col for col in df.columns.to_list() if col.startswith('TMT')]
    df[scol] = df.loc[:, scol].replace(0, np.nan)
    ndf = experiment_assigment(df, exp_pref, usecol)
    if rename_cols is not None:
        _ = channel_exist(df, rename_cols)

    # checking if TMT labels are present in modifications col
    if modifications_checkup(df) is False:
        print('No TMT modifications are present in input table. '
              'Exiting program')
        sys.exit(-1)

    #  --- optional: grouping by experiments for channel renaming
    if isinstance(rename_cols, dict):
        gdfs = renaming_columns(ndf, rename_cols, outname)
        return gdfs
    else:
        ndf.to_csv(f'{outname}_unformatted_PSMs.matrix.tsv',
                   sep='\t', index=False)
        return ndf


def phenodata_prep(filein, outname, args, verbosity):
    df = pd.read_csv(filein, sep=r'\t|\,', header=0, engine='python',
                     na_values=['NA', ''])
    df.columns = df.columns.str.replace('[ |.]', '.', regex=True)
    cols = df.columns.to_list()
    try:
        s = df[['Tag.name', 'Sample.name']]
        equiv = {'Experiment.name': 'Experiment', 'Tag': 'Tag.id',
                 'Sample.name': 'Tag', 'Tag.name': 'Original.Tag'}
        new_names = {col: equiv[col] for col in cols if col in equiv.keys()}
        df.rename(columns=new_names, inplace=True)
        df.to_csv(f'{outname}_formatted_phenodata.tsv', sep='\t',
                  index=False)# na_rep='NA')
    except:
        expected_cols = ['Experiment', 'File.ID', 'Tag',
                         'Peptide.amount']
        intersection = [col for col in df.columns.to_list()
                        if col in expected_cols]
        if len(intersection) == len(expected_cols):
            df.to_csv(f'{outname}_formatted_phenodata.tsv',
                      sep='\t', index=False)  # na_rep='NA')
        else:
            print(f'Minimum expected column names are: {expected_cols}.\n'
                  f'Exiting program')
        sys.exit(-1)
    #  write command
    _ = command_line_out('pheno_data', **args)
    return df


def proteins_prep(filein, file_out, search_engine, args, verbosity):
    df = pd.read_csv(filein, sep='\t', header=0, dtype='object',
                     na_values=['<NA>', ''])
    df.columns = df.columns.str.replace('[ |-]', '.', regex=True)
    df = df.convert_dtypes()
    df.rename(columns={'Number.of.Peptides': 'Number.of.Peptides.total'},
              inplace=True)
    keep = ['Master', 'Protein.Group.IDs', 'Accession', 'Sequence',
            'Contaminant', 'Marked.as', 'Coverage.in.Percent',
            'Number.of.Peptides.total', 'Number.of.PSMs',
            'Number.of.Protein.Unique.Peptides',
            'Number.of.Unique.Peptides',
            'Number.of.AAs', 'MW.in.kDa', 'calc.pI',
            f'Coverage.in.Percent.by.Search.Engine.{search_engine}',
            f'Number.of.PSMs.by.Search.Engine.{search_engine}',
            f'Number.of.Peptides.by.Search.Engine.{search_engine}']

    optional = ['Protein.FDR.Confidence.Combined', 'Exp.q.value.Combined',
                'Biological.Process', 'Cellular.Component', 'Sum.PEP.Score',
                'Molecular.Function', 'Pfam.IDs', 'GO.Accessions',
                'Entrez.Gene.ID', 'Modifications',
                'Number.of.Decoy.Protein.Combined']
    for col in optional:
        if col in df.columns.to_list():
            keep.append(col)

    prot_df = df.loc[:, keep]
    prot_df.to_csv(f'{file_out}_formatted_protein_data.tsv', sep='\t',
                   index=False)#, na_rep='NA')
    #  write command
    _ = command_line_out('protein_data', **args)
    return prot_df


#  ---  nan values ---   #
def nan(df, cols):
    na = 'Number.of.Missing.Values'
    s1 = '.per.Protein.Group'
    s2 = '.per.Peptide'
    if 'Sum.TMT.Abundance' not in df.columns:
        df['Sum.TMT.Abundance'] = df.loc[:, cols].sum(axis=1)
    df[f'{na}'] = df.loc[:, cols].apply(lambda x: x.isna()).sum(axis=1)
    gdf = df.groupby(['Experiment', 'Master.Protein.Accessions'])
    # get min and max NA values per group and populate each entry with them
    df[f'Min.NA{s1}'] = gdf[f'{na}'].transform('min')
    df[f'Max.NA{s1}'] = gdf[f'{na}'].transform('max')
    df[f'Number.of.PSMs{s1}'] = gdf['PSMs.Peptide.ID'].transform('count')
    df[f'Number.of.PSMs.with.MVs{s1}'] = gdf[f'{na}'].transform('sum')
    df['Number.of.Peptides'] = gdf['Sequence'].transform('nunique')
    gdf2 = df.groupby(['Experiment', 'Sequence'])
    df[f'Min.NA{s2}'] = gdf2[f'{na}'].transform('min')
    df[f'Max.NA{s2}'] = gdf2[f'{na}'].transform('max')
    df[f'Number.of.PSMs{s2}.Sequence'] = gdf2['PSMs.Peptide.ID'].transform(
        'count')
    df[f'Number.of.PSMs.with.MVs{s2}.Sequence'] = gdf2[f'{na}'].transform('sum')
    return df

#  ***   End: matrices correction workflows ---

#  ---  Start:  detect existing channels  ---  #


def channel_exist(df, declared_experiments):
    print('the following experiments will have at least one channel renamed:\n',
          '\n'.join(declared_experiments))
    if declared_experiments is not None:
        avail_tmt = list(df.filter(regex=r'^TMT.*\d+').columns)
        decl_tmt = set([k for exp in declared_experiments.keys()
                        for k in declared_experiments[exp]])
        unavail_tmt = [tmt for tmt in decl_tmt if tmt not in avail_tmt]
        if unavail_tmt:
            c = ','.join(unavail_tmt)
            print(f'Error: requested {c} channel(s) not available in psm file.')
            sys.exit(-1)
        else:
            return True
    else:
        return declared_experiments


# --- merging tool  ---   #

def marker_id(df1, df2):
    df1_marker = list(set(df1['marker'].to_list()))
    marker = df2.filter(regex=re.compile('marker',
                                         re.IGNORECASE)).columns.to_list()
    if marker:
        if len(df1_marker) == 1 and df1_marker[0] == 1 and len(marker) == 1:
            del df1['marker']
            df2.rename(columns={marker[0]: 'marker'}, inplace=True)
        return df1, df2
    else:
        return df1, df2


def files_to_merge(filein):
    if filein is not None:
        df = pd.read_csv(filein, sep='\t', header=0, dtype=object)
    else:
        df = pd.DataFrame()
    if not df.empty and 'Accession' not in df.columns.to_list():
        print(f'file {filein} does not contain the [ Accession ] column')
        sys.exit(-1)
    return df


def info_merger(odf1, idf2):
    if not odf1.empty and not idf2.empty:
        df1, df2 = marker_id(odf1, idf2)
        _ = duplicated_columns(df1.columns.to_list(),
                               df2.columns.to_list())
        mdf1 = pd.merge(df1, df2, on='Accession', how='left')
    else:
        mdf1 = odf1
    return mdf1


def eliminate_undesirable_cols(df):
    base_drop = ['level_0', 'level0', 'index']
    pi = df.filter(regex=re.compile('calc.pI',
                                    re.IGNORECASE)).columns.to_list()
    if 'calc.pI_x' in pi and 'calc.pI' not in df.columns.to_list():
        df.rename(columns={'calc.pI_x': 'calc.pI'}, inplace=True)
        pi.remove('calc.pI_x')
    # dropping cols
    to_drop = base_drop + pi
    for col in to_drop:
        try:
            del df[col]
        except:
            pass
    return df


def df_merger(main_input, mf1, af2, outname, verbosity):
    main_df = pd.read_csv(main_input, sep='\t', header=0, dtype=object)
    if mf1 != '':
        markers_df = files_to_merge(mf1)
        extra_df = files_to_merge(af2)
        markers_merged = info_merger(main_df, markers_df)
        merged_df = info_merger(markers_merged, extra_df)
        final_df = eliminate_undesirable_cols(merged_df)
        final_df.rename(columns={'Panther family': 'Annotation'}, inplace=True)
        final_df['marker'] = final_df['marker'].fillna('unknown')
    else:
        extra_df = files_to_merge(af2)
        extra_df.fillna('unknown', inplace=True)
        if 'markers' in extra_df.columns.to_list():
            del extra_df['markers']
        extra_df_subset = extra_df.loc[:, ['Accession', 'tagm.map.allocation',
                                           'tagm.map.probability',
                                           'tagm.map.outlier',
                                           'tagm.map.allocation.cutoff_0.90']]
        final_df = write_mydf([main_df, extra_df_subset],
                            'tagm_added',
                     False, '0.90', '')

    # preparing output file
    dataset = list(set(final_df['Dataset'].to_list()))[0]
    if dataset not in os.path.splitext(outname)[0]:
        fname = f'{dataset}_{os.path.splitext(outname)[0]}'
    else:
        fname = f'{os.path.splitext(outname)[0]}'
    absname = os.path.join(os.getcwd(), fname)
    if mf1 != '':
        final_df.to_csv(f'{absname}.merged.tsv', sep='\t', index=False)
    else:
        final_df.to_csv(f'{absname}.tagm_added.tsv', sep='\t', index=False)
    return final_df


def duplicated_columns(lst1, lst2):
    shared = list(set(lst1).intersection(set(lst2)))
    if shared:
        if 'Accession' in shared:
            shared.remove('Accession')
            if shared:
                dups = ','.join(shared)
                print(f'Duplicated columns in additional dataframe:\n {dups}')
                sys.exit(-1)
            else:
                cols = lst2.remove('Accession')
                return cols
    else:
        if lst2:

            print('Accession column is not in additional dataframe')
            sys.exit(-1)
        else:
            return


def basic_fig_parameters(df, x, y, size, color):
    cols = df.columns.to_list()
    colin = ['marker', 'Annotation']
    ev = len(set(colin).intersection(set(cols)))
    if ev == 2:
        fig = px.scatter(df, x=x, y=y, size=size, color=color, size_max=10,
                         hover_name='Accession',
                         hover_data=['Accession', 'marker', 'Annotation'],
                         custom_data='Annotation')
    elif ev != 2 and 'most.common.pred.SVM.KNN.RF' in cols:
        fig = px.scatter(df, x=x, y=y, size=size, color=color, size_max=10,
                         hover_name='Accession',
                         hover_data=['Accession',
                                     'most.common.pred.SVM.KNN.RF'],
                         custom_data='most.common.pred.SVM.KNN.RF')
    else:
        print('Not enough information provided. '
              'A basic figure will be rendered')
        fig = px.scatter(df, x=x, y=y, size=size, color=color, size_max=10,
                         hover_name='Accession',
                         hover_data=['Accession'],
                         custom_data='Accession')
    return fig


def figure_rendering(filein, x, y, size, color, dim, z='', verbosity=False):
    df = pd.read_csv(filein, sep='\t', header=0, engine='python',
                     encoding='utf-8')
    dim = dim.upper()
    if dim == '3D':
        fig = px.scatter_3d(df, x=x, y=y, z=z, size=size, color=color,
                            size_max=20, hover_name='Accession',
                            hover_data=['Accession', 'marker'])
        bname = os.path.splitext(filein)[0]
        abspath = os.path.join(os.getcwd(), bname)
        fig.write_html(f'{abspath}_{dim}D_plot_{size}_.html',
                       auto_open=True)

    else:
        fig = basic_fig_parameters(df, x, y, size, color)
        bname = os.path.splitext(filein)[0]
        abspath = os.path.join(os.getcwd(), bname)
        fig.write_html(f'{abspath}_{dim}D_plot_{size}_.html',
                       auto_open=False)
        # interactive figure:
        _ = get_my_figure(df, x=x, y=y, size=size, color=color)
    return 'Done'


def args_needed(args):
    if args['figure_dimension'] is None:
        print('dimension was not declared, declare 2D or 3D')
        sys.exit(-1)
    if args['figure_dimension'].lower() == '3d' and args['x_axis'] is not None \
            and args['y_axis'] is not None and args['z_axis'] is not None:
        return
    elif (args['figure_dimension'].lower() == '2d'
          and args['x_axis'] is not None and args['y_axis'] is not None):
        return
    else:
        print('Figure dimension or axis arguments are incomplete')
        sys.exit(-1)


def get_my_figure(df, x, y, size, color):
    '''
    adapted from https://dash.plotly.com/interactive-graphing
    '''
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    fig = basic_fig_parameters(df, x, y, size, color)
    fig.update_layout(clickmode='event+select')

    fig.update_traces(marker_size=10)

    app.layout = html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure=fig
        ),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown("""
                    **Hover Data**
                        Mouse over values in the graph.
                """),
                html.Pre(id='hover-data', style=styles['pre'])
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Click Data**
                    Click on points in the graph.
                """),
                html.Pre(id='click-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Selection Data**
                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.
                    Note that if `layout.clickmode = 'event+select'`, 
                    selection data also
                    accumulates (or un-accumulates) selected data if you hold 
                    down the shift
                    button while clicking.
                """),
                html.Pre(id='selected-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Zoom and Relayout Data**
                    Click and drag on the graph to zoom or click on the zoom
                    buttons in the graph's menu bar.
                    Clicking on legend items will also fire
                    this event.
                """),
                html.Pre(id='relayout-data', style=styles['pre']),
            ], className='three columns')
        ])
    ])

    @callback(
        Output('hover-data', 'children'),
        Input('basic-interactions', 'hoverData'))
    def display_hover_data(hoverData):
        return json.dumps(hoverData, indent=2)

    @callback(
        Output('click-data', 'children'),
        Input('basic-interactions', 'clickData'))
    def display_click_data(clickData):
        return json.dumps(clickData, indent=2)

    @callback(
        Output('selected-data', 'children'),
        Input('basic-interactions', 'selectedData'))
    def display_selected_data(selectedData):
        return json.dumps(selectedData, indent=2)

    @callback(
        Output('relayout-data', 'children'),
        Input('basic-interactions', 'relayoutData'))
    def display_relayout_data(relayoutData):
        return json.dumps(relayoutData, indent=2)

    app.run(debug=True)
    return fig


#  --- subset a df using json data  ---  #


def json_subset(df_in, json_entry):
    df = pd.read_csv(df_in, sep='\t', header=0, engine='python',
                     encoding='utf-8')
    with open(json_entry) as f:
        d = json.load(f)
    subset = pd.json_normalize(d, record_path='points')['hovertext'].to_list()
    ndf = df[~df.Accession.isin(subset)]
    partial_path, suffix = os.path.splitext(df_in)
    outname = os.path.join(partial_path, f'{suffix}.curated.tsv')
    ndf.to_csv(outname, sep='\t', index=False)
    return 'Done'


#   ---  Efficiency calculator   ---   #
def preparing_df(odf):
    # df.columns = df.columns.str.replace(' ', '.')
    ndf = odf.loc[:, ['Annotated.Sequence', 'Modifications', 'Experiment']]
    ndf['Annot.Seq'] = ndf['Annotated.Sequence'].str.extract(
        r'\.([^\.]*)\.', expand=False).str.upper()
    ndf['Modif'] = ndf['Modifications'].str.replace(r'\d+', '',
                                                    regex=True).str.split('; ')
    ndf['Max.modif.per.psm'] = ndf['Modif'].apply(lambda x: len(x) if
                                                  isinstance(x, list) else 0)
    ndf['Modif.unique'] = ndf['Modif'].apply(lambda x: list(set(x))
                                if isinstance(x, list) else 'No modification')
    return ndf


def estimate_modifications(ndf):
    common_modifications = sort_modifications(ndf)
    modifications_per_psm = mods_per_psms(ndf)
    return common_modifications, modifications_per_psm


def mods_per_psms(df):
    sdf = df[~df['Modif.unique'].isin(['No modification'])].copy(deep=True)
    sdf['seq.length'] = sdf['Annot.Seq'].apply(lambda x: len(x))
    sdf['mod.per.psm'] = sdf['Modif'].apply(lambda x: len(x))
    values = {}
    for group, subdf in sdf.groupby('mod.per.psm', as_index=False):
        seq_mean = round(subdf['seq.length'].mean())
        values[group] = [subdf.shape[0], seq_mean]
    ndf = pd.DataFrame.from_dict(values, orient='index',
                                 columns=['counts', 'seq.len.mean'])
    ndf['percentage'] = round((ndf['counts']*100) / sdf.shape[0], 2)
    ndf.sort_index(ascending=True, inplace=True)
    ndf.index.name = 'modif.per.psm'
    ndf['cum.sum'] = ndf['percentage'].cumsum()
    return ndf


def sort_modifications(df):
    df['sorted.mod'] = df['Modif.unique'].apply(lambda x: ', '.join(
        sorted(x)) if isinstance(x, list) else 'No modification')
    sdf = df[~df['sorted.mod'].isin(['No modification'])]
    values = {}
    for group, subdf in sdf.groupby('sorted.mod', as_index=False):
        values[group] = [subdf.shape[0]]
    ndf = pd.DataFrame.from_dict(values, orient='index', columns=['counts'])
    ndf.sort_values(by='counts', ascending=False, inplace=True)
    ndf['percentage'] = round((ndf['counts']*100) / ndf.counts.sum(), 2)
    ndf['cum.sum'] = ndf['percentage'].cumsum()
    ndf.index.name = 'modifications'
    return ndf


def calculating_efficiency(df, exp):
    modifications = df['Modif.unique'].explode().tolist()
    aa_df = aa_modifications(df, modifications)
    dic = {}
    for i in modifications:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    ndf = pd.DataFrame(dic, index=['count']).T
    ndf.index.name = 'modification'
    total_psms = df.shape[0]
    ndf['modif.freq'] = round(ndf['count']*100/total_psms, 2)
    ndf = ndf.sort_values('modif.freq', ascending=False)
    ndf.loc['total PSMs'] = [total_psms, '']
    ndf.reset_index(inplace=True)
    merged = pd.merge(ndf, aa_df, on='modification', how='outer')
    del merged['tupla_mod']
    merged['% efficiency'] = round(merged['count']*100 /
                                   merged['Total.psms.with.aa'], 2)
    try:
        i = merged.loc[merged['modification'] == 'N-Term(TMTpro)'].index[0]
    except:
        i = merged.loc[merged['modification'] == 'N-Term(TMTplex)'].index[0]
    all_psms = merged.loc[merged['modification'] == ('total '
                                                            'PSMs')].index[0]
    merged.loc[i, 'Total.psms.with.aa'] = merged.loc[all_psms, 'count']
    merged.loc[i, '% efficiency'] = merged.loc[i, 'modif.freq']
    merged.fillna('', inplace=True)
    # modifications sorting
    mods = merged['modification'].to_list()
    # sort df by index
    if 'N-Term(TMTpro)' in mods  :
        o = ['N-Term(TMTpro)', 'K(TMTpro)']
    else:
        o = ['N-Term(TMTplex)', 'K(TMTpro)']
    e = [i for i in mods if i not in o]
    sorted_i = o + e
    merged.set_index('modification', inplace=True)
    sorted_df = merged.loc[sorted_i]
    return sorted_df


def aa_modifications(df, modifications):
    exceptions = ['No modification', 'N-Term(TMTpro)', 'N-Term(TMTplex)']
    unique_mods = [mod.replace(')', '').split('(')
                   for mod in set(modifications) if mod not in exceptions]
    aas = {tuple(aa_info): [] for aa_info in unique_mods}
    all_psms = df['Annot.Seq'].to_list()
    for key in aas.keys():
        for psm in all_psms:
            if key[0] in psm:
                aas[key].append(psm)
    ndf = pd.DataFrame.from_dict({k: [len(aas[k])] for k in aas.keys()},
                                 orient='index')
    ndf.index.name = 'tupla_mod'
    ndf.reset_index(inplace=True)
    ndf.rename(columns={0: 'Total.psms.with.aa'}, inplace=True)
    ndf['modification'] = ndf.tupla_mod.apply(lambda x: f"{'('.join(x)})")
    return ndf


def ecalculator(odf, bname):
    psm_df = odf.copy(deep=True).reset_index()
    idf = preparing_df(psm_df)
    for exp, group in idf.groupby('Experiment'):
        common_modif, modifications_per_psms = estimate_modifications(group)
        effic = calculating_efficiency(group, exp)
        with (pd.ExcelWriter(f'{bname}_{exp}.TMT_labeling.efficiency.xlsx')
              as out):
            effic.to_excel(out, sheet_name='Efficiency')
            modifications_per_psms.to_excel(out, sheet_name='Modif per peptide')
            common_modif.to_excel(out, sheet_name='Common modifications')
    print('Efficiency has been calculated')
    return 'Done'

#   ---  shared clusters identification   ---   #


def write_to_excel(dfs_dic, outname):
    with pd.ExcelWriter(f'{outname}.xlsx') as out:
        for key, df in dfs_dic.items():
            df.to_excel(out, sheet_name=key)
    return 'Done'


def clusters_to_dic(odf, label):
    df = odf.copy(deep=True)
    metric = label.split('_')[-2]
    dic = {}
    for clst, group in df.groupby(label):
        if 'unknown' not in clst:
            key = f'{clst}_{metric[:3]}'.replace('cluster', 'clst')
            dic[key] = group['Accession'].to_list()
    return dic


def clst_intersections(dic1, dic2, out):
    dfs_collection = []
    combinations = list(product(list(dic1.keys()), list(dic2.keys())))
    dic1.update(dic2)
    for combination in combinations:
        a, b = sorted(combination)
        intersects = list(set(dic1[a]).intersection(set(dic1[b])))
        d = {'Accession': intersects,
             'clst_size': [int(len(intersects)) for i in
                           range(len(intersects))],
             'clst1': [a for i in range(len(intersects))],
             'clst2': [b for i in range(len(intersects))],
             'new_clst': [(a, b) for i in range(len(intersects))],
             f'shared_cluster.{out}': [f'{a}-{b}' for i
                                       in range(len(intersects))]}
        df = pd.DataFrame(d)
        if not df.empty:
            dfs_collection.append(df)
    fdf = pd.concat(dfs_collection)
    sort_by_acc = sorting_clusters(fdf, 'clst_size', 'Accession')
    tracked_df = tracker_dic(sort_by_acc, out)
    return tracked_df


def tracker_dic(df, out):
    tracker = {}
    for g, sdf in df.groupby(['clst1', 'clst2'], as_index=False):
        size = list(set(sdf['clst_size']))[0]
        clst1, clst2 = list(set(sdf['clst1']))[0], list(set(sdf['clst2']))[0]
        if clst1 not in tracker.keys():
            tracker[clst1] = [(clst1, clst2, size)]
        else:
            if size >= tracker[clst1][0][-1]:
                tracker.update({clst1: [(clst1, clst2, size)]})
        if clst2 not in tracker.keys():
            tracker[clst2] = [(clst1, clst2, size)]
        else:
            if size >= tracker[clst2][0][-1]:
                tracker.update({clst2: [(clst1, clst2, size)]})
    allowed_keys = []
    for k in tracker.keys():
        v = tracker[k][0][:2]
        if not v in allowed_keys:
            allowed_keys.append(v)
    ndf = df.loc[df['new_clst'].isin(allowed_keys)]
    ndf = ndf.loc[:, ['Accession', f'shared_cluster.{out}']]
    return ndf


def sorting_clusters(df, col1, col2):
    df.sort_values(col1, ascending=False, inplace=True)
    best_dfs = []
    for acc, sdf in df.groupby(col2, as_index=False):
        best_dfs.append(sdf.head(1))
    clst_df = pd.concat(best_dfs)
    return clst_df


def get_shared_clusters(odf, label1, label2, out):
    df = odf.copy(deep=True)
    if 'level_0' in df.columns.to_list():
        del df['level_0']
    df.reset_index(inplace=True)
    cls1 = clusters_to_dic(df, label1)
    cls2 = clusters_to_dic(df, label2)
    clst_df = clst_intersections(cls1, cls2, out)
    merged = pd.merge(df, clst_df, on='Accession', how='left')
    columns = ['Dataset', 'Accession', 'calc.pI',
               'Number.of.PSMs.per.Protein.Group']
    merged_columns = merged.columns.to_list()
    inters = set(merged_columns).intersection(columns)
    if inters and len(inters) == len(columns):
        indexes = zip(merged['Dataset'], merged['Accession'],
                      merged['calc.pI'],
                      merged['Number.of.PSMs.per.Protein.Group'])
        merged.index = pd.MultiIndex.from_tuples(indexes, names=columns)
        for i in columns:
            del merged[i]
        return merged, clst_df
    else:
        return merged, clst_df


def clusters_across_datasets(dfs_list):
    starter = [dfs_list[0]]
    to_add = dfs_list[1:]
    counter = -1
    for dfin in to_add:
        counter += 1
        cols = starter[counter].filter(regex=r'^shared_cluster'
                                       ).columns.to_list()
        if len(cols) > 1:
            col1 = cols[-1]
        else:
            col1 = cols[0]
        merged = pd.merge(starter[counter], dfin, on='Accession', how='outer')
        col2 = dfin.filter(regex=r'^shared_cluster'
                           ).columns.to_list()[0]
        suffix = f'{col1}+{col2}'.replace('shared_cluster.', '')
        ndf, pdf = get_shared_clusters(merged, col1, col2, suffix)
        starter.append(ndf)
    all_shared_clsts = starter[-1]
    outname = 'Shared_clusters_across_dataset_comparisons.tsv'
    all_shared_clsts.to_csv(f'{outname}', sep='\t', index=False)
    return all_shared_clsts


def write_mydf(dfs_list, outname, hdbscan, cutoff, accessory_file):
    df = reduce(lambda left, right: pd.merge(left, right,
                                             on='Accession',
                                             how='left'), dfs_list)
    fill_cols = ['SVM.prediction.threshold', 'KNN.prediction.threshold',
                 'RF.prediction.threshold', 'NB.prediction.threshold']
    # compute shared predictions and prepare final df
    pre_final_df = common_prediction(df, fill_cols, hdbscan, cutoff)
    # merge accessory file if passed
    if isinstance(accessory_file, pd.DataFrame):
        acc_cols = [col for col in accessory_file.columns.to_list()
                    if col != 'Accession']
        shared_cols = [col for col in pre_final_df.columns.to_list()
                       if col in acc_cols]
        if len(shared_cols) != 0:
            print(f'Merge will avoid shared columns {shared_cols} between '
                  'master_df and accessory file.\n')
            merge_cols = ['Accession'] + [col for col in acc_cols if
                                          col not in shared_cols]
            final_df = pd.merge(pre_final_df,
                                accessory_file.loc[:, merge_cols],
                                on='Accession', how='left')
        else:
            final_df = pd.merge(pre_final_df, accessory_file,
                                on='Accession', how='left')
    else:
        final_df = pre_final_df

    # write final df
    if 'Dataset' in final_df.columns.to_list():
        dataset = '_' + final_df['Dataset'].to_list()[0]
    else:
        dataset = ''
    fpath = os.path.join(os.getcwd(), f'Final_df{dataset}.{outname}.'
                                      f'Supervised.ML.tsv')
    if outname != 'tagm_added':
        final_df.to_csv(fpath, sep='\t', index=False)
    return final_df


def common_prediction(df, cols, hdbscan=False, cutoff=''):
    ndf = df.copy(deep=True)
    all_cols = ndf.columns.to_list()

    if hdbscan is True:
        if 'hdb_labels_euclidean_TMT_pred_marker' in all_cols:
            cols = cols + ['hdb_labels_euclidean_TMT_pred_marker']
        if 'hdb_labels_euclidean_TMT_pred_marker' in cols:
            ml, size, sign = 'hdbscan', 5, '+hdbscan'
        else:
            ml, size, sign = '', 4, '+'
    else:
        # check if tagm predictions are available
        tagm = list(set([x.split('.')[0] for x in all_cols if
                         x.split('.')[0].startswith('tagm')]))
        if tagm:
            ml, size, sign = 'tagmap', 5, '+tagmap'
            cols = cols + [f'tagm.map.allocation.cutoff_{cutoff}']
        # calculate without hdbscan or tagm predictions
        else:
            ml, size, sign = '', 4, '+'

    ndf.set_index('Accession', inplace=True)
    wdf = ndf.loc[:, cols]
    dic1, dic2, dic3, dic4 = {}, {}, {}, {}
    for acc in wdf.index:
        y = wdf.loc[acc, :].values.flatten().tolist()
        temp = sorted(Counter(y).items(), key=lambda x: x[1], reverse=True)
        if len(temp) == 1:
            dic1[acc] = temp[0][0]
            dic2[acc] = temp[0][1]
            if temp[0][1] >= 3:
                if temp[0][1] >= 4:
                    dic4[acc] = temp[0][0]
                    dic3[acc] = temp[0][0]
                else:
                    if acc not in dic3.keys():
                        dic3[acc] = temp[0][0]
        elif 1 < len(temp) < size:
            if temp[0][1] > temp[1][1]:
                dic1[acc] = temp[0][0]
                dic2[acc] = temp[0][1]
                if temp[0][1] >= 3:
                    if temp[0][1] >= 4:
                        dic4[acc] = temp[0][0]
                        dic3[acc] = temp[0][0]
                    else:
                        if acc not in dic3.keys():
                            dic3[acc] = temp[0][0]
            if size == 5:
                if temp[0][1] == temp[1][1] == temp[2][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0], temp[2][0]])
                    dic2[acc] = 0
                elif temp[0][1] == temp[1][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0]])
                    dic2[acc] = 0
            elif size == 4:
                if temp[0][1] == temp[1][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0]])
                    dic2[acc] = 0
        else:
            dic1[acc] = 'unknown'
            dic2[acc] = 0
    if size == 5:
        df['most.common.pred.SVM.KNN.RF.NB.hdbscan'] = df['Accession'].map(dic1)
    else:
        df['most.common.pred.SVM.KNN.RF.NB'] = df['Accession'].map(dic1)

    complete = [f'most.common.pred.SVM.KNN.RF.NB.{ml}',
                f'most.common.pred.supported.by{sign}',
                f'best.pred.supported3{sign}marker',
                f'best.pred.supported4{sign}marker']
    dics = [dic1, dic2, dic3, dic4]

    for col, dic in zip(complete, dics):
        df[col] = df['Accession'].map(dic)
        df[col] = df[col].fillna('unknown')
    df[complete] = df.loc[:, complete].fillna('unknown')
    return df


def experiments_exist(df1, df2, df1_type, df2_type):
    if not df2.empty:
        df1_exps = list(df1['Experiment'].unique())
        df2_exps = list(df2['Experiment'].unique())
        exp_intersect = set(df1_exps).intersection(set(df2_exps))
        if len(df1_exps) == len(df2_exps) == len(exp_intersect):
            pass
        else:
            print('There is incongruence in the experiment declaration'
                  'among the psms and the pheno files\nExperiments '
                  f'declared are: {df1_type}: {df1_exps}\n{df2_type}: '
                  f'{df2_exps}\nExiting program...')
            sys.exit(-1)
    return

def tmt_sorted_df(df, tmt_channels):
    df_cols = df.columns.tolist()
    tmt_cols = list(df.filter(regex=r'^TMT.*\d+').columns)
    remaining_cols = [col for col in df_cols if col not in tmt_cols] # ordered
    col_by_exp = {}
    for col in tmt_cols:
        label, exp = col.split('.')
        if not exp in col_by_exp.keys():
            col_by_exp[exp] = [label]
        else:
            col_by_exp[exp].append(label)
    # sorting according to pre-set channel order
    new_order = []
    for exp in col_by_exp.keys():
        for label in tmt_channels:
            if label in col_by_exp[exp]:
                chname = f'{label}.{exp}'
                new_order.append(chname)
            else:
                pass
    ordered_cols = remaining_cols + new_order
    sorted_df = df[ordered_cols]
    return sorted_df