import os
import gc
import sys
import numpy as np
import lopit_utils
import pandas as pd
import seaborn as sns
import missingno as msno
import patchworklib as pw
import matplotlib.pyplot as plt


def heatmap(df, outname, suffix):
    color = sns.color_palette('Blues', as_cmap=True)
    if 'Ave.SN' in df.columns:
        new_df = df.loc[:, df.columns != 'Experiment'].sort_values(
            'Ave.SN', ascending=False)
    else:
        new_df = df.loc[:, df.columns != 'Experiment']
    new_df = new_df.apply(lambda x: np.log((x+1)))
    ftitle= 'TMT channel intensity and missing data'
    hmap = heatmap_param(new_df, 4, 3, 'TMT channel',
                         f'{outname}-{suffix}_PSM-index',
                         ftitle, color, True)
    return new_df, hmap


def heatmap_param(df, width, height, x_lab, y_lab, title, cpallete, bar):
    ax1 = pw.Brick(figsize=(width, height))
    sns.heatmap(df, cmap=cpallete, cbar=bar, ax=ax1)
    ax1.set(xlabel=x_lab, ylabel=y_lab)
    ax1.set_title(title)
    return ax1


def barplot(df, outname, suffix):
    ax0 = pw.Brick(figsize=(3, 2))
    nans = df.loc[:, df.columns != 'Ave.SN'].apply(
        lambda x: ((x.isnull().sum().sum()) * 100) / df.shape[0])
    nandf = pd.DataFrame(nans).rename(columns={0: 'Percentage'})
    nandf.reset_index(inplace=True)
    nandf.rename(columns={'index': 'TMT Channels'}, inplace=True)
    sns.set_theme(style='whitegrid')
    sns.barplot(x='TMT Channels', y='Percentage', data=nandf,
                hue='TMT Channels', dodge=False, ax=ax0)
    ax0.set(title=f'Missing Values per TMT channel (%) - {outname}-{suffix}')
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)
    ax0.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    ax0.legend(fontsize='6', title_fontsize='10', markerscale=0.5)
    return ax0


def misva(odf, outname, suffix):
    df = odf.copy(deep=True)
    #   ---  Matrix   ---   #
    ax0 = pw.Brick(figsize=(10, 5))
    msno.matrix(df, ax=ax0)
    msg = f'MISSING VALUES PER TMT CHANNEL - {outname}-{suffix}'
    ax0.set_title(msg)

    #   ---  Bar plot   ---   #
    ax1 = barplot(df, outname, suffix)

    #   ---  Correlation heatmap   ---   #
    newdf = df.loc[:, df.columns != 'Ave.SN']
    ax2 = pw.Brick()
    msno.heatmap(newdf, labels=False, figsize=(8, 6), ax=ax2)
    msg += 'value near -1: if one variable appears then the other '
    msg += 'variable is very likely to be missing.\n'
    msg += 'value near 0: means there is no dependence between the '
    msg += 'occurrence of missing values of two variables.\n'
    msg += 'value near 1: if one variable appears then the other '
    msg += 'variable is very likely to be present.\n'
    ax2.set_title(msg)

    #   ---  Hierarchical correlation dendrogram  ---   #
    ax3 = pw.Brick(figsize=(10, 5))
    msno.dendrogram(newdf, ax=ax3)
    ax3.set_title(f'MVals_cluster_tree - {outname}-{suffix}')
    # plt.savefig(f'{outname}.MVals_cluster_tree.pdf')
    return ax0, ax1, ax2, ax3


def PairGrid(odf, outname):
    df = odf.copy(deep=True)
    df = df.apply(lambda x: np.log2((x + 1)))
    df.dropna(axis=1, how='all', inplace=True)
    g = sns.PairGrid(df, dropna=True, aspect=1, height=1, corner=True)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    g.fig.suptitle(f'{outname}', ha='center',
                   va='bottom', fontsize=14, y=0.95)
    plt.savefig(f'{outname}.PairWise.scatterplot.png', dpi=90)
    fpath = os.path.abspath(f'{outname}.PairWise.scatterplot.png')
    plt.cla()
    plt.clf()
    plt.close()
    return fpath


def scatterplot(df, cols, outname, suffix):
    mycols = cols + ['Ave.SN']
    subset = df.loc[:, mycols].copy(deep=True)
    newname = f'Non-scaled_data-{outname}-{suffix}'
    i1 = PairGrid(subset, newname)
    subset = subset.apply(lambda x: x/subset.sum(axis=1))
    newname = f'Scaled-data-by_Sum.TMT.Abundance-{outname}-{suffix}'
    i2 = PairGrid(subset, newname)
    _ = lopit_utils.merge_images([i1, i2],
                     f'Comparative_PairWise.scatterplot-{outname}-{suffix}')
    return 'Done'


def psm_with_mv(df):
    cols = ['Experiment', 'Master.Protein.Accessions',
            'Master.Protein.Descriptions', 'Min.NA.per.Protein.Group']
    ndf = df.loc[:, cols].copy(deep=True)
    #   ---    filter PSMs with at least one missing value or more   ---   #
    xdf = pd.DataFrame(ndf.groupby(cols[:3], as_index=True
                                   )[cols[3]].apply(lambda x: x > 0))
    xdf.reset_index(inplace=True)
    sub_df = xdf.loc[xdf['Min.NA.per.Protein.Group'].isin([True])]
    s = ndf.loc[sub_df['PSMs.Peptide.ID']]
    s.to_csv('Protein.groups.with.PSMs.withMVs.tsv', sep='\t', index=True,
             na_rep='NA')
    print(f'***   There are {s.shape[0]} protein groups with assigned '
          f'PSMs containing missing values   ***\n{s}')
    return 'Done'


def absval(val, summ):
    a = np.round(val/100 * summ, 0)
    return f'{val:.2f}%\n{round(a)}'


def subplots(df, new_col,  title, outname, suffix):
    all_plots = {}
    for i, group in df.groupby('Experiment', as_index=False):
        if new_col == 'Has.MV':
            group['status'] = ['No missing values', 'Missing values']
        else:
            group['status'] = ['No. of PSMs per protein group > 1',
                               'No. of PSMs per protein group = 1']

        # patch due to pandas version 1.5.3 -> problem with lambda and autopct
        # group.set_index('status', inplace=True)
        sizes = group[f'{new_col}s'].to_list()
        ax0 = pw.Brick(figsize=(11, 6))
        plt.pie(sizes, labels=group['status'].to_list(), startangle=90,
                colors=['#BDE5FD', '#8457FD'], autopct='%1.1f%%')
        # end of patch
        # original code eliminated due to patch:
        # group.set_index('status', inplace=True)
        # ndf = group.loc[:, f'{new_col}s']
        # ax0 = pw.Brick(figsize=(11, 6))
        # ndf.plot.pie(autopct=lambda x: absval(x, ndf.sum()),
        #              explode=[0.05]*2, colors=['#BDE5FD', '#8457FD'],
        #              ax=ax0)
        # end ----   #
        ax0.set_ylabel('')
        ax0.set_title(f'{title} - {outname} MV pie-{i} {suffix}')
        if i not in all_plots.keys():
            all_plots[i] = [ax0]
        else:
            all_plots[i].append(ax0)
    key_order = sorted(all_plots)
    return key_order, all_plots


def daframe_formatting(odf, gcol, cond_col, ncol):
    df = odf.loc[:, gcol + [cond_col]].copy(deep=True)
    if ncol == 'Has.MV':
        df[ncol] = df.groupby(gcol,
                              group_keys=False)[cond_col].apply(lambda x: x > 0
                                                                )
    else:
        df[ncol] = df.groupby(gcol,
                              group_keys=False)[cond_col].apply(lambda x: x == 1
                                                                )
    #   ---
    cols = gcol + [ncol]
    tmp = df.loc[:, cols].copy(deep=True)
    ndf = pd.DataFrame(tmp.groupby([gcol[0], ncol])[gcol[1]].unique())
    ndf[f'{ncol}s'] = ndf.loc[:, gcol[1]].apply(lambda x: len(x))
    ndf.reset_index(inplace=True)
    ndf = ndf.loc[:, [gcol[0], f'{ncol}s']]
    return ndf


def magic_pies(df, gcol, cond_col, new_col, title, outname, suffix):
    print('entering magic pies')
    ndf = daframe_formatting(df, gcol, cond_col, new_col)
    key_order, figs_dic = subplots(ndf, new_col, title, outname, suffix)
    return key_order, figs_dic


def pie_baker(ordered_keys, max_pies, min_pies, new_minpies, psms_pies, suffix):
    zipped_pies = {}
    for g in ordered_keys:
        zipped_pies[g] = [(max_pies[g] + min_pies[g] +
                           new_minpies[g] + psms_pies[g],
                           f'Pies_of_missing_values-{g}-{suffix}')]
    all_paths = []
    for k in zipped_pies.keys():
        all_paths.extend(lopit_utils.compare_loop(zipped_pies[k]))
    _ = lopit_utils.merge_pdfs(all_paths,
                               f'Comparative_pie_charts_'
                               f'of_missing_values-{suffix}.pdf')
    return 'done'


def pie_maker(odf, suffix):
    df = odf.copy(deep=True)
    df.set_index('PSMs.Peptide.ID', inplace=True)
    print('Making pies')
    title = 'Fraction of protein groups containing missing values'
    sk, max_pies = magic_pies(df, ['Experiment', 'Master.Protein.Accessions'],
                              'Max.NA.per.Protein.Group',
                              'Has.MV', title, 'Max', suffix)
    title = 'Fraction of protein groups with all PSMs containing missing values'
    sk, min_pies = magic_pies(df, ['Experiment', 'Master.Protein.Accessions'],
                              'Min.NA.per.Protein.Group',
                              'Has.MV', title, 'Min', suffix)
    _ = psm_with_mv(df)
    title = 'Fraction of protein groups identified with a single PSM'
    sk, nmin_pie = magic_pies(df, ['Experiment', 'Master.Protein.Accessions'],
                              'Number.of.PSMs.per.Protein.Group',
                              'Has.single.PSM', title,
                              'NewMin', suffix)
    fil_df = df[df['Number.of.PSMs.per.Protein.Group'] == 1]
    title = 'Missing values in single-PSM protein groups'
    sk, psm = magic_pies(fil_df, ['Experiment', 'Master.Protein.Accessions'],
                         'Number.of.Missing.Values',
                         'Has.MV', title, 'PSM', suffix)
    _ = pie_baker(sk, max_pies, min_pies, nmin_pie, psm, suffix)
    return 'Pies are ready'


def misvals_removal(df, cols):
    cond =((df['Min.NA.per.Peptide'] == 0) &
           (df['Number.of.Missing.Values'] > 0))
    ndf = df.loc[~cond].copy(deep=True)
    na = 'Number.of.Missing.Values'
    s1 = '.per.Protein.Group'
    s2 = '.per.Peptide'
    s3 = 'Number.of.PSMs'
    todrop = [f'{na}', f'Min.NA{s1}', f'Max.NA{s1}', f'{s3}{s1}',
              f'{s3}.with.MVs{s1}', 'Number.of.Peptides',
              f'Min.NA{s2}', f'Max.NA{s2}', f'{s3}{s2}.Sequence',
              f'{s3}.with.MVs{s2}.Sequence']
    ndf.drop(todrop, axis=1, inplace=True)
    mv_removed = lopit_utils.nan(ndf, cols)
    return mv_removed


def summary(df, kind):
    cols1 = ['Experiment', 'Master.Protein.Accessions']
    cols2 = [col for col in df.columns if col.startswith('TMT')]
    ndf = df.loc[:, [cols1[0]] + cols2]
    mvals = ndf.groupby('Experiment',
                        as_index=True).apply(lambda x: x.isna().sum())
    del mvals['Experiment']
    gvals = df.loc[:, cols1].groupby(
        'Experiment', as_index=True)[f'Master.Protein.Accessions'].apply(
                                                    lambda x: len(x.unique()))
    newgvals = pd.DataFrame(gvals)
    ndf = pd.merge(mvals, newgvals, on='Experiment', how='outer')
    ndf['time-point'] = kind
    ndf.rename(columns={'Master.Protein.Accessions':
                        'Master.Protein.Accessions (groups)'}, inplace=True)
    ndf.reset_index(inplace=True)
    return ndf


def name_assigment(lst1, lst2, lst3, lst4, lst5, prefix):
    figs = [(lst1, f'Comparative_heatmaps_{prefix}'),
            (lst2, f'Comparative_mvals_matrices_{prefix}'),
            (lst3, f'Comparative_mvals_boxplots_{prefix}'),
            (lst4, f'Comparative_mv_correlation_heatmap_{prefix}'),
            (lst5, f'Comparative_mv_cluster_tree_{prefix}')]
    _ = lopit_utils.compare_loop(figs)
    return 'Done'


def misva_bulk_figures(df, suffix):
    heatmaps, matrices, boxplots, hmc, tree = [], [], [], [], []
    for i, group in df.groupby('Experiment', as_index=False):
        group.dropna(axis=1, how='all', inplace=True)
        tmt = list(group.filter(regex=r'^TMT.*\d+').columns)
        _ = scatterplot(group, tmt, i, suffix)  # do not append
        ndf, fig = heatmap(group, i, suffix)
        heatmaps.append(fig)
        mv_matrix, mv_bxp, hmap, dendro = misva(ndf, i, suffix)
        boxplots.append(mv_bxp)
        hmc.append(hmap)
        tree.append(dendro)
    #  write files
    _ = name_assigment(heatmaps, matrices, boxplots, hmc, tree, suffix)
    return 'Done'


def run_heatmap_explorer(file_in, outname):
    print('\n*** - Beginning of mv removal workflow - ***\n')
    dfs_dir = lopit_utils.create_dir('Step3_',
                                     f'DF_ready_for_mv_imputation_{outname}')
    abpath = os.path.abspath(dfs_dir)
    if isinstance(file_in, pd.DataFrame):
        df = file_in
    else:
        df = pd.read_csv(file_in, sep='\t', header=0, engine='python')
        # patch to eliminate blank spaces in quantitative data (bug in PD3.1)
        tmt_cols_prep = list(df.filter(regex=r'^TMT.*\d+').columns)
        for col in tmt_cols_prep:
            df[col] = df[col].replace(' ', '')
            # df[col] = df[col].astype('float')

    newdf = df.copy(deep=True)

    selcol = ['Experiment', 'Average.Reporter.SN']
    tmt = list(newdf.filter(regex=r'^TMT.*\d+').columns)
    newdf1 = newdf.loc[:, tmt + selcol].rename(columns={selcol[1]: 'Ave.SN'})

    #   --- missing value plots   ---   #
    directory = lopit_utils.create_dir('Step3_',
                                       f'Missing_data_figures_{outname}')
    os.chdir(directory)
    #  drawing heat maps, barplots, scatterplots, pies, etc for missing values
    print('Preparing first batch of figures ...')
    _ = misva_bulk_figures(newdf1, 'pre')  # use renamed df '''
    _ = pie_maker(newdf, 'pre')  # use original deep copy

    #   ---   df sizes before missing data removal  ---   #
    print(f'total PSMs in df before mv removal: {newdf.shape[0]}')
    print(f'PSMs by {newdf.groupby(["Experiment"]).size()}')

    # #   ---  removing missing values from original df (no imputation) ---   #
    my_outname = 'Filtered_df-ready-for-imputation'
    filepath = os.path.join(abpath, my_outname)
    rdf = misvals_removal(newdf, tmt)

    #  dataframe to be used for mv imputation in future steps
    rdf.to_csv(f'{filepath}.tsv', sep='\t', index=False)
    sys.exit(-1)
    #   ---  Checking left missing values  ---   #
    ndf = rdf.copy(deep=True)
    newrdf = ndf.loc[:, tmt + selcol].rename(columns={selcol[1]: 'Ave.SN'})

    _ = misva_bulk_figures(newrdf, 'post')  # use renamed df
    # do not do pies on new df-> they won't likely pass the conditionals in f(x)

    print('Preparing summaries for missing values before and after removal.')
    frame1 = summary(newdf, 'Before_MV_removal')
    frame2 = summary(rdf, 'After_MV_removal')
    frames = pd.concat([frame1, frame2])
    oname = 'Total_MV_by_TMT_channel_and_total_protein_groups.tsv'
    frames.to_csv(oname, sep='\t', index=False)
    print('\n*** - MV removal workflow has finished - ***\n')
    os.chdir('..')
    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')
    return rdf


#   ---  Execute   ---   #
if __name__ == '__main__':
    '''
    python <matrix.filtered> <outname>
    matrix.filtered = df_PL1-PL2.tsv or df_PLO-PLN.tsv or PL1PL2_PLNO.tsv
    'D:\PycharmProjects\LOPIT\Step2__First_filter_2023621_162311_SequesHT_TK\df_PL1-PL2.tsv'
    'D:\PycharmProjects\LOPIT\Step2__First_filter_2023621_162311_SequesHT_TK\df_PLO-PLN.tsv'
    'D:\PycharmProjects\LOPIT\Step2__First_filter_2023621_162311_SequesHT_TK\df_PL12-PLNO.tsv'
    '''
    filtered_df_pre_imputation = run_heatmap_explorer(sys.argv[1], sys.argv[2])



