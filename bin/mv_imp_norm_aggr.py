import os
import re
import gc
import sys
import lopit_utils
import pandas as pd
from venn import venn as venny
from scipy.stats import gmean
import psm_diagnostics as dia
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


channels = ['TMT126',
            'TMT127N', 'TMT127C', 'TMT128N', 'TMT128C',
            'TMT129N', 'TMT129C', 'TMT130N', 'TMT130C',
            'TMT131N', 'TMT131C', 'TMT132N', 'TMT132C',
            'TMT133N', 'TMT133C', 'TMT134N', 'TMT134C',
            'TMT135N']

#   ---  missing values imputation  ---   ###

def imputeMinDet(odf, q, cols):
    gdf = odf.copy(deep=True)
    gdf.set_index(['PSMs.Peptide.ID', 'Sum.TMT.Abundance'], inplace=True)
    print('Imputing missing values with MinDet on channels', cols)
    dfcols = list(gdf.filter(regex=r'^TMT.*\d+').columns)
    redef_cols = [col for col in dfcols if col in cols]
    remaining_cols = [col for col in dfcols if
                      col not in redef_cols]
    gdfs = []
    # slicing columns of interest to calculate Min Det for the subgroup #
    slice = gdf.loc[:, cols]
    remaining_slice = gdf.loc[:, remaining_cols]
    quantiledic = dict(zip(slice.columns, slice.quantile(q, axis=0)))
    # imputing mv in the subgroup
    imputed_subgroup = slice.fillna(quantiledic)
    reconstited_df = pd.merge(remaining_slice, imputed_subgroup,
                              left_index=True, right_index=True)
    gdfs.append(reconstited_df)
    imputed_cat = pd.concat(gdfs)
    imputed_cat.reset_index(inplace=True)
    return imputed_cat


def imputeknn(df, cols, k, exp):
    print('Imputing missing values with knn on channels ', cols)
    gdf = df.copy(deep=True)
    subdfcols = list(gdf.filter(regex=r'^TMT.*\d+').columns)
    gdf.set_index('PSMs.Peptide.ID', inplace=True)
    redef_cols = [col for col in subdfcols if col in cols]
    remaining_cols = [col for col in subdfcols if
                      col not in redef_cols]
    # slicing sdfs
    gdfcols = gdf.loc[:, redef_cols]
    remaining_slice = gdf.loc[:, remaining_cols]
    remaining_slice.reset_index(inplace=True)

    if not isinstance(gdfcols, pd.DataFrame):  # if only 1 column was passed (?)
        gdfcols = pd.DataFrame(gdfcols)

    # scaling data by division by 'Sum.TMT.Abundance' prior to imputation
    gdfcols2 = gdfcols.apply(lambda x: x/gdf['Sum.TMT.Abundance'])

    #   ---   knn imputation   ---   #
    imputer = KNNImputer(n_neighbors=k)
    imp_df = pd.DataFrame(imputer.fit_transform(gdfcols2),
                          columns=gdfcols2.columns,
                          index=gdfcols2.index.to_list())
    imp_df.index.name = 'PSMs.Peptide.ID'
    # unscaling data by multiplication by 'Sum.TMT.Abundance' ---   #
    imp_df_copy = imp_df.copy(deep=True)
    imp_df_copy = imp_df_copy.apply(lambda x: x * gdf['Sum.TMT.Abundance'])
    imp_df_copy = imp_df_copy.round(1)
    imp_df_copy.reset_index(inplace=True)
    #   return unscaled df with imputed data  ---  #
    gdfcols.reset_index(inplace=True)
    gdfcols = gdfcols.fillna(imp_df_copy)
    knn_imp_cat = pd.merge(remaining_slice, gdfcols, on='PSMs.Peptide.ID',
                           how='outer')
    knn_imp_cat['Experiment'] = exp
    return knn_imp_cat


def compare_df(df1, df2, cols, exp, verbosity):
    ndf1 = df1.loc[:, cols].copy(deep=True)
    ndf1.set_index('PSMs.Peptide.ID', inplace=True)
    ndf2 = df2.loc[:, cols].copy(deep=True)
    ndf2.set_index('PSMs.Peptide.ID', inplace=True)
    imp = ndf1.compare(ndf2)
    if verbosity:
        imp.to_csv(f'Diff.imputation{exp}.tsv', sep='\t', index=True)
    return 'Done'


#   ---  Data normalization   ---   ###



def corr_by_peptide_amount(df1, df2):
    new = df1.copy(deep=True)
    sel = list(new.filter(regex='^TMT').columns)
    exps = new['Experiment'].unique().tolist()
    #   ---   peptide amount correction ---
    #   ---   calculation of the correction value   ---   #
    ave_exp = pd.DataFrame(df2.groupby('Experiment')[
                                  'Peptide.amount'].mean()).reset_index()
    ave_exp.rename(columns={'Peptide.amount': 'Peptide.amount.mean'},
                   inplace=True)
    minidf = df2.loc[:, ['Experiment', 'Tag', 'Peptide.amount']]
    merged = pd.merge(minidf, ave_exp, on='Experiment', how='outer')
    merged['corr.val'] = merged['Peptide.amount']/merged['Peptide.amount.mean']
    m_ave = merged.loc[merged.Experiment.isin(exps)]
    m_ave = m_ave.loc[:, ['Experiment', 'Tag', 'corr.val']]
    nm_ave = pd.pivot_table(m_ave, index='Experiment', columns='Tag')
    nm_ave.columns = nm_ave.columns.get_level_values(1)
    nm_ave.reset_index(inplace=True)
    #   ---   tmp df to hold correction values   ---   #
    df1_ids = new.loc[:, ['PSMs.Peptide.ID', 'Experiment']].copy(deep=True)
    ndf = df1_ids.merge(nm_ave, on='Experiment', how='outer')
    #   ---  performing  correction   ---   #
    odf = new.loc[:, sel]
    div = odf/ndf.loc[:, sel]
    new.update(div)  # updated full df
    return new


def corr_by_peptide_loading(df):
    new = df.copy(deep=True)
    cols = list(new.filter(regex='^TMT').columns)
    slice = new.loc[:, ['Experiment'] + cols]
    num = slice.groupby('Experiment').sum()
    num['denominator'] = num.mean(axis=1)
    corr = num.loc[:, cols].div(num['denominator'], axis=0)
    corr.reset_index(inplace=True)
    m = pd.merge(slice.loc[:, 'Experiment'], corr, on='Experiment', how='outer')
    div = slice.loc[:, cols]/m.loc[:, cols]
    new.update(div)
    return new


def normalization(df):
    cols = list(df.filter(regex='^TMT').columns)
    new = df.copy(deep=True)
    new['Sum.TMT.Abundance'] = new.loc[:, cols].sum(axis=1)
    corr = new.loc[:, cols].div(new['Sum.TMT.Abundance'], axis=0)
    new.update(corr)
    return new

    #   ---  PSMs aggregation   ---   ###


def read_prot_class(filein, search_engine):
    df = pd.read_csv(filein, sep='\t', header=0, dtype='object',
                     na_values=['<NA>', ''])
    df.columns = df.columns.str.replace('[ |-]', '.', regex=True)
    df = df.convert_dtypes()
    df.rename(columns={'Number.of.Peptides': 'Number.of.Peptides.total'},
                  inplace=True)
    keep = ['Protein.FDR.Confidence.Combined', 'Master',
            'Protein.Group.IDs',
            'Accession', 'Sequence', 'Exp.q.value.Combined', 'Contaminant',
            'Marked.as', 'Sum.PEP.Score',
            'Number.of.Decoy.Protein.Combined',
            'Coverage.in.Percent', 'Number.of.Peptides.total', 'Number.of.PSMs',
            'Number.of.Protein.Unique.Peptides',
            'Number.of.Unique.Peptides',
            'Number.of.AAs', 'MW.in.kDa', 'calc.pI',
            f'Coverage.in.Percent.by.Search.Engine.{search_engine}',
            f'Number.of.PSMs.by.Search.Engine.{search_engine}',
            f'Number.of.Peptides.by.Search.Engine.{search_engine}',
            'Biological.Process',
            'Cellular.Component', 'Molecular.Function', 'Pfam.IDs',
            'GO.Accessions', 'Entrez.Gene.ID', 'Modifications']
    return df.loc[:, keep]


def flatten(x):
    try:
        lval = list(set([e.strip() for i in x for e in i.split(' ')]))
        val = '; '.join(lval)
        return len(lval), val
    except:
        newx = 1*x
        return 1, newx


def flatten_colnames(df):
    fun = '<lambda>'
    tmt = {k: k[0] for k in df.columns.to_list() if k[0].startswith('TMT')}
    gp = {k: k[0] for k in df.columns.to_list() if k[0].endswith('Group')}
    tmt.update(gp)
    ncols = {
        ('Master.Protein.Accessions', ''): 'Master.Protein.Accessions',
        (f'Protein.Accessions', f'{fun}'): 'Protein.Accessions',
        (f'Sequence', f'{fun}'): 'Sequence',
        ('Protein.Descriptions', f'{fun}'): 'Protein.Descriptions'}

    ncols.update(tmt)
    df.columns = df.columns.to_flat_index()
    df.rename(columns=ncols, inplace=True)
    return df


def aggregation_split(df, col1, col2, verbosity):
    df[['Number.of.Proteins', col1]] = pd.DataFrame(df[col1].tolist(),
                                                    index=df.index)
    df[['Number.of.Peptides', col2]] = pd.DataFrame(df[col2].tolist(),
                                                    index=df.index)
    #   ---  eliminating column created in step above   ---   #
    df.reset_index(inplace=True)
    if verbosity:
        df.to_csv('First_agg.tsv', sep='\t', index=False)
    return df


def aggregation(df, dfprot, taginf, verbosity):
    #   ---  columns to retain   ---   #
    regcol = df.filter(regex='.per.Protein.Group').columns.to_list()
    tmt = df.filter(regex='^TMT').columns.to_list()
    keep = ['Experiment', 'Master.Protein.Accessions', 'Sequence',
            'Master.Protein.Descriptions', 'Protein.Accessions',
            'Protein.Descriptions', 'Number.of.Proteins',
            'Number.of.Peptides']
    keep.extend(regcol)
    keep.extend(tmt)
    #   ---  sliced df with retained columns   ---   #
    newdf = df.loc[:, keep].copy(deep=True)

    #   --- dic of functions for aggregation by desired column ----   #
    strs_cols = {**{k: [lambda x: flatten(x)] for k in [keep[4], keep[2]]},
                 **{k: [lambda x: flatten(x)[1]] for k in [keep[5]]},
                 **{k: 'median' for k in tmt},
                 **{k: 'mean' for k in regcol}}

    #   ---   aggregation by dic info   ---   #
    grouped = newdf.groupby(['Master.Protein.Accessions', 'Experiment'])
    agg_df = grouped.agg(strs_cols)

    #   ---  changing col names originated by aggregation   ---   #
    agg_df = flatten_colnames(agg_df)

    #   ---  splitting the aggregated columns by their flatten tuple  ---   #
    split_df = aggregation_split(agg_df, keep[4], keep[2], verbosity)
    newnames = {'Master.Protein.Accessions': 'Accession',
                'Master.Protein.Descriptions': 'Description',
                'Sequence': 'Peptide.Sequences'}
    split_df.rename(columns=newnames, inplace=True)
    # merged = pd.merge(split_df, dfprot, on='Accession', how='inner')
    merged = guess_data_type(split_df, dfprot)
    #   ---   combine feature by median   ---   #
    merged.sort_values('Experiment', ascending=True)
    if verbosity:
        merged.to_csv('median.tsv', sep='\t', index=False)
    if len(set(merged.Experiment.tolist())) > 1:
        _ = venn(merged)
    renormalized = normalization(merged)
    renormalized.index.name = 'PSMs.Peptide.ID'
    renormalized.reset_index(inplace=True)
    return renormalized


def merge_and_irs(exp, df_list, verbosity):
    #  --- plex aggregation by psms   #
    merged_dup = df_list[0].merge(df_list[1], left_on='Accession',
                                  right_on='Accession', how='inner')
    merged = merged_dup.T.drop_duplicates().T
    bridge_cols = merged.filter(regex='^TMT131C').columns.to_list()
    tmt = [col for col in merged.filter(regex='^TMT').columns.to_list()
           if col not in bridge_cols]
    #  --- irs correction   ---   #
    irs_df = internal_reference_scaling(merged, tmt, bridge_cols)
    irs_df['Experiment'] = exp
    if verbosity:
        irs_df.to_csv(f'irs_corrected_{exp}.tsv', sep='\t', index=True)
    return irs_df


def reconstitute_plexes(exp, dfs_list, verbosity):
    irs_df = merge_and_irs(exp, dfs_list, verbosity)
    toadd = irs_df.filter(regex='PSMs.per.Protein.Group').columns.to_list()
    newcol = 'Number.of.PSMs.per.Protein.Group'
    irs_df[newcol] = irs_df.loc[:, toadd].sum(axis=1)
    irs_df.reset_index(inplace=True)
    kept_tmt = irs_df.filter(regex='^TMT').columns.to_list()
    other_cols = [col for col in irs_df.columns.to_list()
                  if col not in kept_tmt]
    sorted_cols = sorted_channels(kept_tmt, other_cols)
    irs_df_sorted = irs_df.loc[:, sorted_cols].copy(deep=True)
    if verbosity:
        irs_df_sorted.to_csv(f'Normalized.df.agg.post-irs-{exp}.tsv', sep='\t')
    return irs_df_sorted


def new_heads(no_tmt_cols):
    names = {}
    for col in no_tmt_cols:
        ncol = '.'.join(col.split('.')[:-1])
        if ncol != '':
            if ncol not in names:
                names[ncol] = [col]
            else:
                names[ncol].append(col)
    return {names[k][0]: k for k in names.keys()
            if len(names[k]) == 1}


def internal_reference_scaling(df, tmt_cols, bridge_cols):  # hard coded
    # testcols = tmt_cols[:4]
    cols_1 = [col for col in tmt_cols if col.endswith('1')
              and col not in bridge_cols]
    cols_2 = [col for col in tmt_cols if col.endswith('2')
              and col not in bridge_cols]
    df_copy = df.copy(deep=True).set_index('Accession')
    new_df = df_copy.loc[:, bridge_cols].astype('float')
    new_df['geometric_mean'] = gmean(new_df, axis=1)
    factors = ['irs_factor_1', 'irs_factor_2']
    new_df[factors] = new_df.loc[:,  bridge_cols].apply(
                                lambda x: new_df['geometric_mean'] / x)
    # correction for odd fractions (1)
    irs_1 = df_copy.loc[:, cols_1].apply(lambda x: new_df['irs_factor_1'] * x)
    # correction for even fractions (2)
    irs_2 = df_copy.loc[:, cols_2].apply(lambda x: new_df['irs_factor_2'] * x)
    # merging both sub-dfs ---  #
    merged = pd.merge(irs_1, irs_2, left_index=True,
                      right_index=True, how='inner')
    # updating original df --
    updated = merged.combine_first(df_copy).reindex(df_copy.index)
    # updated df and remove bridge columns ---
    cols = [col for col in updated.columns.tolist() if col not in bridge_cols]
    updated_df = updated.loc[:, cols]
    no_tmt_cols = [col for col in cols if not col.startswith('TMT')]
    updated_df.rename(columns=new_heads(no_tmt_cols), inplace=True)
    return updated_df


def label_names(labels_list):
    if len(labels_list) == 4:
        labs = sorted(labels_list)
        labs = ('_'.join(labs[0:2]), '_'.join(labs[2:]))
    else:
        labs = labels_list
    return labs


def venn(df):
    labels = tuple(set(df.Experiment.tolist()))
    exp_dics = {}
    for i, group in df.groupby('Experiment')['Accession']:
        exp_dics[i] = set(group.tolist())
    fig, ax = plt.subplots()  # Get the figure and axes objects
    venny(exp_dics, ax=ax)  # Pass the axes object to venn

    # Attempt to change font size
    for text in ax.texts:
        text.set_fontsize(5)  # Set desired font size

    plt.savefig(f"VennDiagram{'-'.join(labels)}.pdf")
    plt.clf()
    plt.close()
    return 'Done'


def sorted_channels(current_tmtcols, other):
    tmts = [tmt.split('.')[0] for tmt in current_tmtcols]
    exps = sorted(list(set([tmt.split('.')[-1] for tmt in current_tmtcols])))
    tmt_cols = []
    for exp in exps:
        for channel in channels:
            if channel in tmts:
                tmt_cols.append(f'{channel}.{exp}')
    sorted_cols = other + tmt_cols
    return sorted_cols



def channels_by_exp(df):
    exp_tmt = {}
    df_copy = df.copy(deep=True)
    for e, sdf in df_copy.groupby('Experiment', as_index=False):
        sdf.dropna(axis=1, how='all', inplace=True)
        exp_tmt[e] = list(sdf.filter(regex=r'^TMT.*\d+').columns)
    return exp_tmt


def check_key(k):
    if '-' in k:
        return list(set(k.split('-')))
    else:
        return [k]


def declared_exp_tmt(imp, tmt_by_exp, method_name):
    to_impute = {}
    if imp is not None:
        for k in imp.keys():
            exps = check_key(k)
            if isinstance(exps, list):
                for entry in exps:
                    to_impute[entry] = imp[k]
            else:
                to_impute[k] = imp[k]
    for k in tmt_by_exp.keys():
        if k not in to_impute.keys():
            to_impute[k] = []
            print(f'*** WARNING!: only one imputation method for {k} '
                  f'was declared, see details below ***')
    return to_impute


def remainder_tmt(avail_tmt_exp, value_list):
    return list(set(avail_tmt_exp).difference(set(value_list)))


def remove_empty_keys(dic):
    new_dic = {}
    for k in dic.keys():
        if k not in new_dic.keys():
            new_dic[k] = {}
            for key in dic[k].keys():
                if dic[k][key]:
                    new_dic[k][key] = dic[k][key]
        else:
            if dic[k][key]:
                new_dic[k][key].append(dic[k][key])
    return new_dic


def catch_missing_channels(dic, tot_chan):
    exit_program = []
    all_params = remove_empty_keys(dic)
    for k in all_params.keys():
        channs = [e for v in dic[k].values() for e in v]
        if not channs:
            exit_program.append(k)
        else:
            missing_channs = set(tot_chan[k]).difference(channs)
            if missing_channs:
                print(f'there is/are not declared imputation method'
                      f'for some channels in exp {k}')
                sys.exit(-1)
    if exit_program:
        exps = ', '. join(exit_program)
        print(f'ERROR: at least one imputation method must be declared '
              f'for {exps}\n')
        sys.exit(-1)
    return all_params


def validate_imputation_params(tmt_by_exp, mnar_name, method1,
                               mar_name, method2):
    dic = {}
    for k in tmt_by_exp.keys():
        if k not in dic.keys():
            dic[k] = {}
            dic[k][mnar_name] = list(set(method1[k]))
            if not method2[k] and not method1[k]:
                dic[k][mar_name] = []
            if 'remainder' in method2[k] and not method1[k]:
                dic[k][mar_name] = remainder_tmt(tmt_by_exp[k],
                                                 method1[k])
            if 'remainder' in method2[k] and method1[k]:
                dic[k][mar_name] = remainder_tmt(tmt_by_exp[k],
                                                 method1[k])
            if not method2[k] and method1[k]:
                dic[k][mar_name] = remainder_tmt(tmt_by_exp[k],
                                                 method1[k])
    validated_dic = catch_missing_channels(dic, tmt_by_exp)
    return validated_dic


def imputation(odf, imp_params, verbosity):
    sel_cols = ['Experiment'] + list(odf.filter(regex=r'^TMT.*\d+').columns)
    acc_info = odf.loc[:, ~odf.columns.isin(sel_cols)]
    dfs = []
    df = odf.copy(deep=True)
    for exp, sdf in df.groupby('Experiment', as_index=False):
        print(f'imputing mv for {exp}')
        sdf.dropna(axis=1, how='all', inplace=True)
        sdfcols = list(sdf.filter(regex=r'^TMT.*\d+').columns)

        #   --- MinDet   ---   #
        try:
            mindet_list = imp_params[exp]['MinDet']
        except:
            mindet_list = []
            print(f'WARNING: no MinDet imputation carried out for exp {exp}')
        if mindet_list:
            imputed1 = imputeMinDet(sdf, 0.01, mindet_list)
        else:
            sdf_copy = sdf.copy(deep=True)
            slices = ['PSMs.Peptide.ID', 'Sum.TMT.Abundance'] + sdfcols
            imputed1 = sdf_copy.loc[:, slices]

        # ---   knn   ---   #
        try:
            knn_list = imp_params[exp]['knn']

        except:
            print(f'ERROR: no MinDet and knn imputation carried out '
                  f'for exp {exp}')
            sys.exit(-1)
        if knn_list:
            imputed2 = imputeknn(imputed1, knn_list, 10, exp)
            mycols = ['Experiment', 'PSMs.Peptide.ID'] + mindet_list + knn_list
            _ = compare_df(sdf, imputed2, mycols, exp, verbosity)
            if verbosity:
                imputed2.to_csv(f'imputed.{exp}.MinDet-knn.tmp.tsv',
                                sep='\t', index=False)
            dfs.append(imputed2)

        else:
            print('Unknown reason triggers else statement in imputation')
            imputed1.to_csv(f'imputed.{exp}.MinDet.only.tmp.tsv',
                            sep='\t', index=False)
            dfs.append(imputed1)

    rec = pd.concat(dfs)
    new_df = pd.merge(rec, acc_info, on='PSMs.Peptide.ID', how='inner')
    return new_df


def channel_exists(chan_by_exp, req_chan):
    print('Checking that requested channels exist in experiments')
    collect = {}
    req_chans = {k: req_chan[key] for key in req_chan.keys()
                 for k in check_key(key)}
    for key in req_chans.keys():
        if 'remainder' in req_chans[key]:
            reqs = [v for v in req_chans[key] if v != 'remainder']
            difference = set(reqs).difference(chan_by_exp[key])
        else:
            difference = set(req_chans[key]).difference(chan_by_exp[key])
        if difference:
            collect[key] = list(difference)
    if collect:
        for k in collect.keys():
            print(f'Existing channels in {k} are:\n', chan_by_exp[k])
            print('Requested channel(s) for MV imputation but missing '
                  f'from {k} is/are:\n', collect[k])
            print('Exiting program...')
            sys.exit(-1)
    return 'done'


def param_verification(psms, chan_mnar, mnar, chan_mar, mar):
    print('Checking imputation parameters...')
    tmt_by_exp = channels_by_exp(psms)
    # verify if chan_nmar are exist in df
    verify_chan_mnar = channel_exists(tmt_by_exp, chan_mnar)
    # verify if chan_mar are exist in df
    verify_chan_mar = channel_exists(tmt_by_exp, chan_mar)
    # continue workflow
    method_mnar_dic = declared_exp_tmt(chan_mnar, tmt_by_exp, mnar)
    method_mar_dic = declared_exp_tmt(chan_mar, tmt_by_exp, mar)
    imputation_params = validate_imputation_params(tmt_by_exp, mnar,
                                                   method_mnar_dic,
                                                   mar, method_mar_dic)
    return imputation_params


def get_boxplot(df, taginf, outname, mytitle, psms=True, reconstituted=False):
    if reconstituted:
        boxplot_path = dia.catplot(df, taginf, mytitle)
    else:
        mypivot_df = dia.large_pivot_tab(df, taginf, outname,
                                         True, psms,
                                         reconstituted)
        boxplot_path = dia.catplot(mypivot_df, taginf, mytitle)

    return boxplot_path


def df_splitter(odf, reconstitute, verbosity):
    if reconstitute is not None:
        inv_dic = {v: k for k in reconstitute.keys() for v in reconstitute[k]}
        exps_reconst = [v.strip() for val in reconstitute.values() for v in val]
    else:
        inv_dic = {}
        exps_reconst = []
    reconst_dfs = {}
    donot_reconst_dfs = {}
    for exp, sdg in odf.groupby('Experiment', as_index=False):
        cols = [col for col in sdg.columns.to_list() if col != 'Accession']
        if exp in exps_reconst:
            # updating headers prior to reconstitution
            expstorec = reconstitute[inv_dic[exp]]
            dic = catch_uncommon_ends(expstorec)
            sdg.rename(columns={f'{i}': f'{i}.{dic[exp]}' for i in cols},
                       inplace=True)
            if inv_dic[exp] not in reconst_dfs.keys():
                reconst_dfs[inv_dic[exp]] = [sdg]
            else:
                reconst_dfs[inv_dic[exp]].append(sdg)
        else:
            ncols = {f'{i}': f'{i}.{exp}' for i in cols if i != 'Experiment'}
            sdg.rename(columns=ncols, inplace=True)
            psmsid = sdg.filter(regex='PSMs.Peptide.ID').columns.to_list()
            for psms in psmsid:
                del sdg[psms]
            donot_reconst_dfs[exp] = sdg
    # writing aggregated pre-irs dfs
    if verbosity:
        for i in reconst_dfs.keys():
            for v in reconst_dfs[i]:
                v.to_csv(v.to_csv(f'Normalized.df.agg.pre-irs-{i}.tsv',
                                  sep='\t'))
    return donot_reconst_dfs, reconst_dfs


def catch_uncommon_ends(exp_list):
    matching_string = os.path.commonprefix(exp_list)
    r = re.compile(matching_string)
    dic = {}
    for entry in exp_list:
        unmatched_string = r.sub('', entry)
        suffix = f'{matching_string[-1]}{unmatched_string}'
        dic[entry] = suffix
    return dic


def reconstitute_validation(df, reconstitute):
    exps_reconst = [v.strip() for val in reconstitute.values() for v in val]
    all_exps = list(set(df['Experiment'].to_list()))
    valid = [exp for exp in exps_reconst if exp in all_exps]
    if len(exps_reconst) != len(valid):
        exps = set(all_exps).difference(set(exps_reconst))
        print(f'Experiments {exps} are absent in input data or \n'
              f'experiment declaration is command line wrong \nExiting program')
        sys.exit(1)
    return 'Done'


def plots_for_irs_correction(working_plexes):
    # # ---  box plot post IRS correction   ---  #
    figs_paths = []
    for exp in working_plexes.keys():
        working_plex_df = working_plexes[exp].copy(deep=True)
        exp = list(set(working_plex_df['Experiment']))[0]
        irs_df1, irs_df2, tag1, tag2 = dia.irs_pivot_table(working_plex_df,
                                                           '')
        irsdf1_path = get_boxplot(irs_df1, tag1, 'irs.1',
                                  f'TagAbunbyExp.post-IRS-{exp}.1',
                                  False, True)
        irsdf2_path = get_boxplot(irs_df2, tag2, 'irs.1',
                                  f'TagAbunbyExp.post-IRS-{exp}.2',
                                  False,True)
        figs_paths.append(irsdf1_path)
        figs_paths.append(irsdf2_path)
    return figs_paths


def reconstitution_manager(odf, reconstitute, verbosity):
    no_recons_dic, reconst_dic = df_splitter(odf, reconstitute, verbosity)
    reconstituted_plexes_dic = {}
    for key in reconst_dic.keys():
        working_plex = reconstitute_plexes(key, reconst_dic[key], verbosity)
        reconstituted_plexes_dic[key] = working_plex
    print('****** Dataframe reconstitution has finished')
    return no_recons_dic, reconstituted_plexes_dic


def guess_data_type(df1, df2):  # guessing if accession only or accession-psm
    df1_accs = list(set(df1['Accession'].to_list()))
    df2_accs = df2['Accession'].to_list()
    intersection = set(df1_accs).intersection(df2_accs)
    if len(intersection) == len(df1_accs): # df1 must be encompassed in df2
        merged = pd.merge(df1, df2, on='Accession', how='inner')
        return merged
    else:
        df1['tmp'] = df1['Accession'].apply(lambda x: x[0: x.rfind('_')])
        df2.rename(columns={'Accession': 'Accession_orig'}, inplace=True)
        merged = pd.merge(df1, df2, left_on='tmp', right_on='Accession_orig',
                          how='inner')
        merged.drop(['tmp', 'Accession_orig'], axis=1, inplace=True)
        return merged


def imp_agg_normalize(psmfile, pheno_file, protein_file,
                      fileout, cmds, mnar, channels_mnar,
                      mar, channels_mar, reconstitute, verbosity):
    print('\n*** - Beginning of mv imputation and df aggregation workflow - ***')
    #   ---   Creating main dir   ---   #
    mydir = lopit_utils.create_dir('Step4_',
                                   f'PSM_Normalization_{fileout}')
    os.chdir(mydir)
    #   --- working dataframes ---   #
    if isinstance(psmfile, pd.DataFrame):
        pre_parsed_psm = psmfile
    else:
        pre_parsed_psm = pd.read_csv(psmfile, sep='\t', header=0,
                                     engine='python')
    if pheno_file is not None:
        pre_parsed_pheno = pd.read_csv(pheno_file, sep=r'\t|\,', header=0)
    else:
        pre_parsed_pheno = pd.DataFrame()

    pre_parsed_prots = pd.read_csv(protein_file, sep='\t', header=0)

    #   ---  obtaining custom tags
    taginf = lopit_utils.custom_taginf(lopit_utils.taginf, pre_parsed_psm)
    #   ---  writing commands to a file   ---   #
    cmd_line = lopit_utils.command_line_out('imp_agg', **cmds)
    #   ---   checking recontitution parameters   ---   #
    if reconstitute:
        _ = reconstitute_validation(pre_parsed_psm, reconstitute)

    #   ---  checking mv imputation parameters --- #
    imputation_params = param_verification(pre_parsed_psm, channels_mnar, mnar,
                                           channels_mar, mar)
    # print('Imputation parameters are:\n', imputation_params)
    #   ---   mv imputation   ---   #
    print('Imputation of missing values')
    imputed_df = imputation(pre_parsed_psm, imputation_params, verbosity)

    # TMT columns need to be sorted **************************************************************

    #   ---   Data correction   ---   #

    if not pre_parsed_pheno.empty:
        # checking phenodata experiment declaration and experiments in PSMs file
        _ = lopit_utils.experiments_exist(imputed_df, pre_parsed_pheno,
                                          'imputed psms',
                                          'pheno')
        print('\n*-- Correction using peptide amount AND sample loading '
              'will be done --*')
        first_correction = corr_by_peptide_amount(imputed_df, pre_parsed_pheno)
        second_correction = corr_by_peptide_loading(first_correction)
    else:
        print('\n*-- Correction using ONLY peptide loading will be done --*')
        second_correction = corr_by_peptide_loading(imputed_df)

    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')

    #  boxplot for corrections
    g1 = get_boxplot(second_correction, taginf, 'my_pivot.1',
                     'TagAbunbyExp.corrected')

    #  Normalization by dividing PSMs intensities over Sum.TMT.Abundance
    normalized = normalization(second_correction)

    #  boxplot for normalization after correction
    g2 = get_boxplot(normalized, taginf, 'my_pivot.2',
                     'TagAbunbyExp.norm')
    if verbosity:
        normalized.to_csv('Main_normalized.df.preagg.tsv', sep='\t',
                          index=False)

    #   ---   psms aggregation   ---   #
    aggregated_df = aggregation(normalized, pre_parsed_prots, taginf, verbosity)
    if verbosity:
        aggregated_df.to_csv('Main_normalized.df.agg.tsv', sep='\t',
                             index=False)

    #   ---   boxplot post-aggregation and pre IRS correction   ---   #

    g3 = get_boxplot(aggregated_df, taginf, 'my_pivot.3',
                     'TagAbunbyExp.norm.agg.pre-IRS')

    plots1 = [g1, g2]

    _ = lopit_utils.merge_images(plots1, f'Comparative_boxplots')

    # --- reconstitute experiment plexes if requested
    non_rec, reconst = reconstitution_manager(aggregated_df,
                                              reconstitute,
                                              verbosity)

    #   ---   dic containing all dfs by exp   ---   #
    all_dfs_dic = {**non_rec, **reconst}
    for experiment in all_dfs_dic.keys():
        all_dfs_dic[experiment].dropna(axis=1, how='all', inplace=True)
        tmt_types = {col: float for col in all_dfs_dic[
                        experiment].columns.to_list() if col.startswith('TMT')}
        all_dfs_dic[experiment].astype(tmt_types, copy=True).dtypes
        tmt_sorted_df = lopit_utils.tmt_sorted_df(all_dfs_dic[experiment],
                                                  lopit_utils.tmt_chans)
        tmt_sorted_df.to_csv(f'df_for_clustering_{experiment}.tsv',
                                       sep='\t', index=False)
    #   ---  all plots ---   #
    print('Creating comparative plots')
    paths_lst = plots_for_irs_correction(reconst)
    if paths_lst:
        plots = [g3] + paths_lst
        _ = lopit_utils.merge_images(plots, f'Comparative_boxplots-'
                                     f'with_reconstitution')
    else:
        _ = lopit_utils.merge_images([g3],
                                     f'Comparative_boxplots_'
                                     f'without-reconstitution')

    print('*** - mv imputation and aggregation workflow has finished - ***\n')
    os.chdir('../..')

    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')
    return list(all_dfs_dic.values())
    #  ---

#   ---   Execute   ---   #
'''
base = 'D:\\PycharmProjects\\LOPIT\\Perkinsus_LOPIT_K\\Data\\Nov_2021_Mascot\'
script.py <arg1> <arg2> <arg3> 
arg1 = D:\PycharmProjects\LOPIT\\filtered_df.pre-imputation.tsv 
arg2 = base + \PL_pData.csv
arg3 = base + kb601_20211127_PmarLOPIT_Mascot_multicon_Proteins.txt
Arg4 = search engine, choose between Mascot or Sequest.HT 
arg5 = write file True
arg6 = reconstitution
df_PL1-PL2.tsv'
PL_pData.csv'
PL12_PLN12_PLO12_Proteins.txt' 

Sequest.HT
False
True
False (for PL12 but True for PLNO)
'''
if __name__ == '__main__':
    pre_imputation_df = sys.argv[1]
    phenodata = sys.argv[2]
    protein_info_file = sys.argv[3]
    outname = sys.argv[4]  # string
    cmds = 'dict_w_commands'
    mnar = sys.argv[5]  # it should be the string 'mnar'
    imp_method1 = sys.argv[6]  # dictionary
    mar = sys.argv[7]   # it should be the string 'mar'
    imp_method2 = sys.argv[8]  # dictionary
    reconstitution = sys.argv[9]  # dictionary
    all_dfs = imp_agg_normalize(pre_imputation_df, phenodata,
                                protein_info_file, outname,
                                cmds, mnar, imp_method1, mar, imp_method2,
                                reconstitution)
