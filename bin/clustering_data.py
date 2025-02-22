import gc
import os
import sys
import glob
import hdbscan
import lopit_utils
import numpy as np
import pandas as pd
from sklearn import metrics
import graphic_results as gr
from functools import reduce
from collections import Counter
from itertools import combinations
# from memory_profiler import profile
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib.colors import hex2color
from natsort import index_natsorted, natsorted
from sklearn.preprocessing import StandardScaler
from lopit_utils import tmt_sorted_df
try:
    import rmm, cudf
    from cuml.manifold import TSNE
    from cuml import UMAP
    from cuml.cluster import HDBSCAN
    from cuml.decomposition import TruncatedSVD
    cuml = True
    sklearn = False
    rmm.reinitialize(pool_allocator=True,
                     managed_memory=True)
except ImportError:
    sklearn = True
    cuml = False
    from sklearn.manifold import TSNE
    from umap import UMAP


#  -- cluster projection parameters ---   #


def create_myloop(perplexity, add_umap, pca, marker_info='', pred=False,):
    if pca:
        param = [('PC1', 'PC2', 'PCA'),
                 (f'tSNE_dim_1_2c_{perplexity}', f'tSNE_dim_2_2c_{perplexity}',
                  'tSNE'),
                 ('UMAP_dim1_2c', 'UMAP_dim2_2c', 'UMAP')]
    else:
        param = [(f'tSNE_dim_1_2c_{perplexity}', f'tSNE_dim_2_2c_{perplexity}',
                  'tSNE'),
                 ('UMAP_dim1_2c', 'UMAP_dim2_2c', 'UMAP')]

    myloop = {('hdb_labels_euclidean_TMT', 'hdb_prob_euclidean_TMT',
               'euclidean'): param,
              ('hdb_labels_manhattan_TMT', 'hdb_prob_manhattan_TMT',
               'manhattan'): param}
    if marker_info != '':
        myloop.update({
            ('marker', 'hdb_prob_euclidean_TMT', 'euclidean'): param,
            ('marker', 'hdb_prob_manhattan_TMT', 'manhattan'): param})
        if pred:
            myloop.update({
                ('hdb_labels_euclidean_TMT_pred_marker',
                 'hdb_prob_euclidean_TMT', 'euclidean'): param,
                ('hdb_labels_manhattan_TMT_pred_marker',
                 'hdb_prob_manhattan_TMT', 'manhattan'): param})

    if add_umap:
        myloop.update({('hdb_labels_euclidean_UMAP', 'hdb_prob_euclidean_UMAP',
                       'euclidean'): param,
                       ('hdb_labels_manhattan_UMAP', 'hdb_prob_manhattan_UMAP',
                      'manhattan'): param})
    return myloop


def create_marker_loop(perplexity):
    param = [('PC1', 'PC2', 'PCA'), (f'tSNE_dim_1_2c_{perplexity}',
                                     f'tSNE_dim_2_2c_{perplexity}', 'tSNE'),
             ('UMAP_dim1_2c', 'UMAP_dim2_2c', 'UMAP')]
    marker_loop = {('marker', 'hdb_prob_euclidean_TMT', 'euclidean'): param,
                   ('marker', 'hdb_prob_manhattan_TMT', 'manhattan'): param,
                   ('marker', 'hdb_prob_euclidean_UMAP', 'euclidean'): param,
                   ('marker', 'hdb_prob_manhattan_UMAP', 'manhattan'): param}
    return marker_loop


features_loop = {
    'psms1': ['quantiles', 'quantiles', (30, 1), 'flare',
              'PSMs (quantiles)'],
    'psms2': ['PSMs_by_prot_group', 'PSMs_by_prot_group', (30, 1),
              'flare', 'PSMs (number)'],
    'pI': ['calc.pI', 'calc.pI', (5, 30), 'flare', 'pI'],
    'signalp': ['SP_ranges', 'SP_ranges', (30, 2), 'flare', 'SP'],
    'targetp': ['TP_ranges', 'TP_ranges', (30, 2), 'flare', 'TP'],
    'tm': ['tmhmm-PredHel', 'tmhmm-PredHel', (2, 30), 'flare', 'TM'],
    'gpi': ['gpi-anchored', 'gpi-anchored', (2, 30), 'flare', 'GPI-A']}


def predominant_marker(value):
    temp = sorted(Counter(value).items(), key=lambda x: x[1], reverse=True)
    temp2 = {tup[0]: tup for tup in temp}
    if len(temp2.keys()) > 1:
        if 'unknown' in temp2:
            temp.remove((temp2['unknown']))
    if len(temp) >= 2:
        if temp[0][1] > temp[1][1]:
            value = temp[0][0]
        else:
            value = f'{temp[0][0]}|{temp[1][0]}'
    else:
        value = temp[0][0]
    return value


def predominant_marker_mmapper(df, hdbscan_colname):
    tmp_dic = {}
    for index, g in df.groupby([hdbscan_colname], group_keys=True)['marker']:
        if index == 'unknown':
            z = 'unknown'
        else:
            z = predominant_marker(g)
        tmp_dic[index] = z
    df[f'{hdbscan_colname}_pred_marker'] = df[hdbscan_colname].map(tmp_dic)
    return df


def create_tsne_sklearn(df, components, method, perplex):
    # ndf = df.copy(deep=True)
    if method == 'barnes_hut':
        theta = 0
        tsne = TSNE(n_components=components, random_state=1661,
                    learning_rate='auto', init='random', perplexity=perplex,
                    method=method, n_jobs=-1, n_iter=5000, angle=theta)
    else:
        tsne = TSNE(n_components=components, random_state=1661,
                    learning_rate='auto', init='random', perplexity=perplex,
                    method=method, n_jobs=-1, n_iter=5000)
    embedding = tsne.fit_transform(df)
    #  --  creating a dataframe with the 2 new dimensions  --  #
    embedding_df = pd.DataFrame()
    embedding_df.index = df.index
    if components == 2:
        embedding_df[f'tSNE_dim_1_2c_{perplex}'] = embedding[:, 0]
        embedding_df[f'tSNE_dim_2_2c_{perplex}'] = embedding[:, 1]
    else:
        embedding_df[f'tSNE_dim_1_3c_{perplex}'] = embedding[:, 0]
        embedding_df[f'tSNE_dim_2_3c_{perplex}'] = embedding[:, 1]
        embedding_df[f'tSNE_dim_3_3c_{perplex}'] = embedding[:, 2]
    #   --  return tsne fitted dataframe   -- #
    return embedding_df


def create_tsne_cuml(df, components, method, perplex):
    # cuml-tsne only supports two components as of Oct-10-23
    print(f'tSNE via cuml')
    cols = df.columns.to_list()
    ndf = df.copy(deep=True).reset_index()
    cdf = ndf.loc[:, cols]
    if method == 'barnes_hut':
        theta = 0.5
        tsne = TSNE(n_components=components, random_state=1155,
                    learning_rate=300, init='random', perplexity=perplex,
                    method=method, n_iter=5000, angle=theta,
                    exaggeration_iter=300,
                    n_neighbors=3*perplex, output_type='pandas')
    else:
        tsne = TSNE(n_components=components, random_state=1155,
                    learning_rate=300.0, init='random', perplexity=perplex,
                    method=method, n_iter=10000, exaggeration_iter=300,
                    n_neighbors=3*perplex, output_type='pandas')

    embedding_df = tsne.fit_transform(cdf)

    #  --  creating a dataframe with the 2 new dimensions  --  #
    embedding_df.rename(columns={0: f'tSNE_dim_1_2c_{perplex}',
                                 1: f'tSNE_dim_2_2c_{perplex}'},
                        inplace=True)
    indexes = zip(ndf['Dataset'], ndf['Accession'],
                  ndf['calc.pI'], ndf['Number.of.PSMs.per.Protein.Group'])
    index_names = ['Dataset', 'Accession', 'calc.pI',
                   'Number.of.PSMs.per.Protein.Group']
    embedding_df.index = pd.MultiIndex.from_tuples(indexes, names=index_names)
    #   --  return tsne fitted dataframe   -- #
    return embedding_df


def create_umap(df, experiment, marker, components, mindistance, n_neighbors):
    # default for mindistance is set from submenu or cmd line
    print(f'UMAP via cuml')
    if n_neighbors is None:
        n_neighbors = int(np.ceil(np.sqrt(df.shape[0])))
    reducer = UMAP(n_components=components, min_dist=mindistance,
                   n_neighbors=n_neighbors, init='random', n_epochs=1000)
    scaled_data = StandardScaler().fit_transform(df)
    embedding = reducer.fit_transform(scaled_data)
    dimensions = [f'UMAP_dim{i +1}_{components}c' for i in range(0, components)]
    embedding_df = pd.DataFrame(embedding, columns=dimensions,
                                index=df.index)

    embedding_df['marker'] = embedding_df.index.map(marker)
    embedding_df['marker'].fillna(0)
    if components == 2:
        umap_plot = gr.plot_umap(embedding_df, experiment, dimensions)
    return embedding_df


# @profile
def my_tsne(df, dataset, marker, components, method, perplex):
    if sklearn:
        tsne_df = create_tsne_sklearn(df, components, method, perplex)
    elif cuml:
        tsne_df = create_tsne_cuml(df, components, method, perplex)
    else:
        print(f'error running tsne, sklearn is {sklearn}, cuml is {cuml}')
        sys.exit(-1)
    tsne_df['marker'] = tsne_df.index.map(marker)
    # tsne_df['marker'].fillna('unknown', inplace=True)
    tsne_df['marker'] = tsne_df['marker'].fillna('unknown')
    if components == 2:
        _ = gr.plot_tsne(tsne_df, dataset, components, perplex)
    return tsne_df


def variance_info(fitted_pca):
    eigenvalues = fitted_pca.explained_variance_  # variance explained by eachPC
    explained_variance_ratio = fitted_pca.explained_variance_ratio_
    variance_ratio_cum = np.cumsum(fitted_pca.explained_variance_ratio_)
    weights = fitted_pca.components_  # represents the elements of eigenvectors
    num_pc = fitted_pca.n_features_in_
    return explained_variance_ratio, weights, num_pc


def pc_99(explained_variance_ratio, pc_list, white_transformed_pc):  # not used
    pc_df = pd.DataFrame(explained_variance_ratio, index=pc_list,
                         columns=['sum'])
    csum = pd.DataFrame(np.cumsum(pc_df, axis=0))  # cumulative sum of pcs
    pc_99 = (csum[csum['sum'] <= 0.99]).index.values  # explaining up to 99%
    whittened_tranformed_pc_99 = white_transformed_pc.loc[:, pc_99]
    return whittened_tranformed_pc_99


def check_numeric(series):
    """Checks if a Pandas Series contains any non-numeric values."""
    for val in series:
        if not isinstance(val, (int, float, np.number)):
            return False
    return True


# @profile
def my_pca(df, dataset, comp):
    '''
    https://builtin.com/machine-learning/pca-in-python
    '''
    print(f'Generating a PCA and selecting {comp} best components')
    newdir = gr.new_dir(f'PCA')
    os.chdir(newdir)
    ndf = df.copy(deep=True)
    if cuml:
        print('PCA via cuml')
        cdf = cudf.DataFrame.from_pandas(ndf)
        # Iterative approach to find optimal n_components
        explained_variance_ratio_ = 0
        n_components = 1
        desired_variance = 0.99
        # `while loop` stops when desired variance is reached or when the
        # number of components is equal the number of features or rows.
        while explained_variance_ratio_ < desired_variance and n_components < min(
                cdf.shape):
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(cdf)
            explained_variance_ratio_ = svd.explained_variance_ratio_.sum()
            n_components += 1

        n_components -= 1  # adjust n_components for the last increment in the loop
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(cdf)
        transformed_data = svd.transform(cdf)
        transformed_df = transformed_data.to_pandas()
        transformed_df.to_csv(f'PCA_results_{dataset}.tsv',
                              sep='\t', index=True)
        os.chdir('..')
        return transformed_df
    else:
        #  getting a whitened PCA and projecting the PCs onto the matrix  #
        pipeline = Pipeline([('scaling', StandardScaler()),
                             ('pca', PCA(n_components=comp, svd_solver='full',
                                         whiten=True))])
        white_transformed_pca = pipeline.fit_transform(ndf)
        # white_transformed_pca = PCA().fit_transform(ndf)
        white_transformed_pc = pd.DataFrame(white_transformed_pca,
                                            index=ndf.index)
        newnames = {i: f'PC{i + 1}' for i in white_transformed_pc.columns.to_list()}
        white_transformed_pc.rename(columns=newnames, inplace=True)
        white_pca = gr.pca_plot(white_transformed_pc, f'{dataset}_whiten')
        # --  getting eigenvalues and variance  ---
        whitened_fit_pca = PCA(n_components=comp,
                               whiten=True).fit(ndf)
        explained_variance_ratio, weights, num_pc = variance_info(whitened_fit_pca)
        pc_list = ['PC' + str(i) for i in list(range(1, num_pc + 1))]
        #  --- scree plot and correlation matrix --- #
        _ = gr.scree_plot(explained_variance_ratio, pc_list, dataset)
        _ = gr.correlation_matrix_plot(ndf, dataset, pc_list, weights)
        white_transformed_pc.to_csv(f'PCA_white_results_{dataset}.tsv',
                                    sep='\t', index=True)
        os.chdir('..')
        return white_transformed_pc


# @profile
def my_umap(dfs, dataset, marker, min_dist, n_neighbors, verbosity):
    df, pca_df = dfs
    # umap for only 2 components
    print(f'UMAP embeddings for 2 components')
    umap_2dim_df = create_umap(df=df,
                               experiment=f'{dataset}_2-dims',
                               marker=marker,
                               components=2,
                               mindistance=min_dist,
                               n_neighbors=n_neighbors)

    # set minimum distance very low (e.g., 5e-324) for density-based clustering
    if pca_df is None:
        print(f'UMAP embeddings for 3 components')
        # was set to pca
        umap_xdim_df = create_umap(df=df,
                                   experiment=f'{dataset}_3-dims',
                                   marker=marker,
                                   components=3,
                                   mindistance=min_dist,
                                   n_neighbors=n_neighbors)
    else:
        print(f'UMAP embeddings for {pca_df.shape[1]} pca components')
        umap_xdim_df = create_umap(df=pca_df,
                                   experiment=f'{dataset}_{pca_df.shape[1]}-dim',
                                   marker=marker,
                                   components=pca_df.shape[1],
                                   mindistance=5e-324,
                                   n_neighbors=n_neighbors)
    del umap_2dim_df['marker']
    del umap_xdim_df['marker']
    umap_all_dims_df = pd.merge(umap_2dim_df, umap_xdim_df, left_index=True,
                                right_index=True, how='inner')
    if verbosity:
        fpath = os.path.join(os.getcwd(), 'UMAP_multiple_dims_df.tsv')
        umap_all_dims_df.to_csv(fpath, sep='\t', index=True)
    return umap_all_dims_df


def adjusted_rand_index(df1, df2, experiment, dftype):
    sub_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    new_df = sub_df[(sub_df[f'hdb_labels_euclidean_{dftype}'] &
                     sub_df[f'hdb_labels_manhattan_{dftype}']) != -1]
    euclidean = new_df[f'hdb_labels_euclidean_{dftype}'].to_list()
    manhattan = new_df[f'hdb_labels_manhattan_{dftype}'].to_list()
    ari = metrics.adjusted_rand_score(euclidean, manhattan)
    df = pd.DataFrame({f'Total proteins in experiments {dftype}':
                           [sub_df.shape[0]],
                       f'Proteins in clusters {dftype}': [new_df.shape[0]],
                       f'ARI {dftype}': [ari],
                       f'No. euclidean clusters {dftype}':
                           [len(list(set(euclidean)))],
                       f'No. manhattan clusters {dftype}':
                           [len(list(set(manhattan)))],
                       f'experiment': [experiment]})
    return df


def hdbscan_workflow(df, dataset, dftype, offset, epsilon, min_size,
                     min_sample):
    newdir = gr.new_dir(f'HDBSCAN')
    os.chdir(newdir)
    cdf1, euc_df, euc_pers, estats, euc_relv = my_hdbscan(df,
                                                          'euclidean',
                                                          dataset, dftype,
                                                          offset, epsilon,
                                                          min_size,
                                                          min_sample)
    cdf2, man_df, man_pers, mstats, manh_relv = my_hdbscan(df,
                                                           'manhattan',
                                                           dataset, dftype,
                                                           offset, epsilon,
                                                           min_size,
                                                           min_sample)
    rel_validity = pd.merge(euc_relv, manh_relv, left_on='experiment',
                            right_on='experiment', how='inner')
    del man_pers['experiment']
    persistence_df = pd.merge(euc_pers, man_pers, left_index=True,
                              right_index=True, how='outer')
    hdbs_stats = pd.concat([estats, mstats])
    sum_df0 = adjusted_rand_index(cdf1, cdf2, dataset, dftype)
    sum_df1 = pd.merge(sum_df0, rel_validity, left_on='experiment',
                       right_on='experiment', how='inner')
    # change cluster labels

    euc_df = change_name(euc_df, 'euclidean', dftype)
    man_df = change_name(man_df, 'manhattan', dftype)
    # merge hdbscan results
    hdb_df = pd.merge(euc_df, man_df, left_index=True,
                      right_index=True, how='inner')
    sum_df1.to_csv(f'Summary_{dataset}_{dftype}.tsv', sep='\t',
                   index=True)
    hdbs_stats.to_csv(f'Hdb_stats_{dataset}_{dftype}.tsv', sep='\t',
                      index=True)
    persistence_df.to_csv(f'Hdb_persistence_df_{dataset}_{dftype}.tsv',
                          sep='\t', index=True)
    os.chdir('..')
    return hdb_df, hdbs_stats, persistence_df, sum_df1


def replacements_dic(df, col):
    onames = list(set(df[col].to_list()))
    nnames = [f'cluster{n+1}' if n != -1 else 'unknown' for n in onames]
    return dict(zip(onames, nnames))


def change_name(df, metric, dftype):
    # rename clusters
    xcol = f'hdb_labels_{metric}_{dftype}'
    clst_rename = replacements_dic(df, xcol)
    df[xcol] = df[xcol].replace(clst_rename)
    return df


# @profile
def my_hdbscan(df, metric, dataset, dftype, offset, epsilon, min_size,
               min_sample):
    ndf = df.copy(deep=True)
    if min_size is None:
        min_size = int(np.floor(np.log(ndf.shape[0])))
    if min_sample is None:
        min_sample = min_size + offset
    # hdbscan clustering using mostly defaults.

    if cuml:
        print('HDBSCAN via cuml')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,
                                    min_samples=min_sample,
                                    cluster_selection_epsilon=epsilon,
                                    max_cluster_size=0,
                                    metric=metric,
                                    alpha=1.0,
                                    p=2,
                                    cluster_selection_method='leaf',
                                    allow_single_cluster=False,
                                    gen_min_span_tree=True,
                                    prediction_data=True)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,
                                    min_samples=min_sample,
                                    metric=metric,
                                    algorithm='best',
                                    leaf_size=40,
                                    allow_single_cluster=False,
                                    prediction_data=True,
                                    alpha=1.0,
                                    cluster_selection_method='leaf',
                                    cluster_selection_epsilon=epsilon,
                                    core_dist_n_jobs=4,
                                    approx_min_span_tree=True,
                                    gen_min_span_tree=True,
                                    match_reference_implementation=False)

    cluster_labels = clusterer.fit_predict(ndf)
    hdbscan_df = pd.DataFrame(zip(clusterer.labels_,
                                  clusterer.probabilities_,
                                  clusterer.outlier_scores_),
                              columns=[
                                  f'hdb_labels_{metric}_{dftype}',
                                  f'hdb_prob_{metric}_{dftype}',
                                  f'hdb_outlier_scores_{metric}_{dftype}'],
                              index=df.index)

    # --
    clusters = hdbscan_df.loc[:, [f'hdb_labels_{metric}_{dftype}']
                              ].copy(deep=True)

    hdbscan_df.to_csv(f'Hdbscan_df_{dataset}_{metric}_{dftype}.tsv',
                      sep='\t',
                      index=True)
    # print('clusterer.relative_validity_', clusterer.relative_validity_)

    clusters_labels = []
    for i in clusterer.labels_:
        if i not in clusters_labels:
            clusters_labels.append(i)
    cluster_persist = pd.DataFrame(zip(clusters_labels,
                                       clusterer.cluster_persistence_),
                                   columns=[f'hdb_labels_{metric}_{dftype}',
                                            f'cluster_persist_{metric}_{dftype}'
                                            ])
    cluster_persist.set_index(f'hdb_labels_{metric}_{dftype}',
                              inplace=True)
    cluster_rel_val = pd.DataFrame({f'hdb_relative_validity_{metric}_{dftype}':
                                   [clusterer.relative_validity_],
                                    'experiment': [dataset]})
    cluster_persistence_stats = pd.DataFrame(cluster_persist.describe())
    cluster_persistence_stats = cluster_persistence_stats.T
    cluster_persistence_stats['experiment'] = dataset
    cluster_persist['experiment'] = dataset

    return clusters, hdbscan_df, cluster_persist, \
        cluster_persistence_stats, cluster_rel_val


def check_label_len(df, label):
    vals = list(df[label].unique())
    vals.remove('unknown')
    vals = natsorted(vals)
    return vals


def custom_color_dic(df, label):
    vals = check_label_len(df, label)
    if len(vals) <= 50:
        custom_hex_colors = lopit_utils.colors[: len(vals)]
        custom_rgb_colors = [hex2color(color) for color in custom_hex_colors]
        color_dic = dict(zip(vals, custom_rgb_colors))
        color_dic.update({'unknown': hex2color('#cab2d6')})
        return color_dic
    else:
        print('Number of markers surpasses the color pallete in lopit_utils')
        sys.exit(-1)


def projections(odf, dataset, sizes, loop_dic):
    if not isinstance(odf, pd.DataFrame):
        df = pd.read_csv(odf, sep='\t', header=0)
        dataset = list(set(df['Dataset'].to_list()))[0]
    else:
        df = odf

    for i in loop_dic.keys():
        labels, probs, distance = i

        for v in loop_dic[i]:
            coord1, coord2, dirname = v
            if not os.path.isdir(dirname):
                gr.new_dir(dirname)
            os.chdir(dirname)
            df.sort_values(by=[labels],
                           key=lambda x: np.argsort(
                               index_natsorted(df[labels])),
                           ascending=True, inplace=True)
            custom_palette = custom_color_dic(df, labels)
            _ = gr.projection(df, coord1, coord2, labels, probs, sizes,
                              dataset, custom_palette,
                              f'{dirname}_{distance}{labels}')
            print('cwd', os.getcwd())
            os.chdir('..')
    return 'Done'


def df_integration(df1, df2, dataset):
    #   ---   integrating charms into main df  ---   #
    df_in1 = pd.merge(df1, df2, on='Accession', how='left')
    #   ---   calculating quantiles and psms sums by prot group ---   #
    ndf = psms_sums_by_protgroup(df_in1)
    for i in ['level_0', 'level0', 'index']:
        if i in ndf.columns.to_list():
            del ndf[i]
    print(f'Your final file containing TMT and accessory data integrated is:\n'
          f'Final_df_{dataset}.tsv')
    fpath = os.path.join(os.getcwd(), f'Final_df_{dataset}.tsv')
    ndf.to_csv(fpath, sep='\t', index=False)
    return ndf


def feature_projections(df, coord1, coord2, dataset, c_type, dic):
    cols = df.columns.to_list()
    updated_dic = {k: dic[k] for k in dic.keys() if dic[k][0] in cols}
    for i in updated_dic.keys():
        labels, probs, sizes, palette, outname = updated_dic[i]
        os.chdir(c_type)
        pj = gr.projection(df, coord1, coord2, labels, probs, sizes,
                           dataset, palette, f'{c_type}_{outname}')
        os.chdir('..')
    return 'Done'


def create_dic_df(files_list):
    if isinstance(files_list, str):
        files_list = [files_list]
    my_dfs = {}
    for entry in files_list:
        if isinstance(entry, pd.DataFrame):
            full_df = entry.copy(deep=True)
        else:
            if os.path.isfile(entry):
                full_df = pd.read_csv(entry, header=0, sep='\t')
            else:
                print(f'Input {entry} is not a file or dataframe')
                sys.exit(-1)

        experiment = list(set(full_df['Experiment'].to_list()))[0]
        cols = kept_columns(full_df)
        if experiment not in my_dfs.keys():
            my_dfs[experiment] = full_df.loc[:, cols]
        else:
            print(f'something is wrong with the input file:\n {entry}')
    return my_dfs


def new_psms(df):
    c = df.filter(regex='^Number.of.PSMs.per.Protein.Group').columns.to_list()
    newc = [i for i in c if '_' not in i]
    psms = df.copy(deep=True)
    psms.set_index('Accession', inplace=True)
    psms = psms.loc[:, newc]
    gdf = pd.DataFrame(psms.sum(1),
                       columns=['Number.of.PSMs.per.Protein.Group'])
    gdf.reset_index(inplace=True)
    return newc, gdf


def multi_indexing_df(df, dataset):
    subc = list(set([col.rsplit('.')[-1] for col in df.columns.to_list()]))
    e = [col for col in subc if (col != 'Accession') and (col != 'Dataset')][0]
    if e == dataset:
        n = {f'calc.pI.{e}': 'calc.pI',
             f'Number.of.PSMs.per.Protein.Group.{e}':
                 'Number.of.PSMs.per.Protein.Group'}
        df.rename(columns=n, inplace=True)
    #   ---  leave only tmt cols for clustering methods  ---   #
    index_names = ['Dataset', 'Accession', 'calc.pI',
                   'Number.of.PSMs.per.Protein.Group']
    indexes = zip(df['Dataset'], df['Accession'],
                  df['calc.pI'], df['Number.of.PSMs.per.Protein.Group'])

    df.index = pd.MultiIndex.from_tuples(indexes, names=index_names)
    tmt = df.filter(regex='^TMT').columns.to_list()
    for col in df.columns.to_list():
        if col not in tmt:
            del df[col]
    return df


def dataset_to_dic(dic):
    newdic = {}
    for experiment in dic.keys():
        df_copy = dic[experiment].copy(deep=True)
        df_copy['Dataset'] = experiment
        miindex_df = multi_indexing_df(df_copy, experiment)
        newdic[experiment] = miindex_df
    return newdic


def dataset_grouping(dfs_dic, combos, dstype):
    comb_dic = {''.join(comb): comb for comb in combos}
    if dstype == 'all':
        mixes = dataset_to_dic(dfs_dic)
    else:
        subset_dic = {k: dfs_dic[k] for k in comb_dic.keys()
                      if k in dfs_dic.keys()}
        mixes = dataset_to_dic(subset_dic)
    for c in comb_dic.keys():
        if c not in mixes.keys():
            print(f'Grouping Dataset: {c}')
            dfs = [dfs_dic[k] for k in dfs_dic.keys() if k in comb_dic[c]]
            if not dfs:
                print(f"No dataframes found for {c}. Please check value "
                      f"declared contains experiment groups separated by '-' "
                      f"and multiple groups are separated by ','.\n"
                      f"Exiting program...")
                sys.exit(-1)
            mix = reduce(lambda left, right: pd.merge(left, right,
                                                      on='Accession',
                                                      how='inner'),
                         dfs)
            no_dups = mix.T.drop_duplicates().T
            pi = no_dups.filter(regex='^calc.pI').columns.to_list()
            calc_pi = {k: '.'.join(k.split('.')[:-1])
                       if len(k.split('.')) > 2 else k.split('_')[0] for k in pi}
            no_dups.rename(columns=calc_pi, inplace=True)
            #  --- updating PSMs values ---   #
            torem, new_psms_values = new_psms(no_dups)
            torem += no_dups.filter(regex='_').columns.to_list()
            #   ---  removing extra columns   ---
            for entry in torem:
                del no_dups[entry]
            new_no_dups = pd.merge(no_dups, new_psms_values,
                                   on='Accession', how='inner')
            new_no_dups['Dataset'] = c
            multi_index_df = multi_indexing_df(new_no_dups, c)
            mixes[c] = multi_index_df
    return mixes


def kept_columns(df):
    calc_pi = df.filter(regex='^calc.pI').columns.to_list()
    psms_pg = df.filter(regex='^Number.of.PSMs.per.Protein.Group'
                             ).columns.to_list()
    tmt = df.filter(regex='^TMT').columns.to_list()
    cols = ['Accession'] + calc_pi + psms_pg + tmt
    return cols


def create_combinations(experiments_list):
    ranges = [key + 1 for key in range(1, len(experiments_list) + 1)]
    tmp_set = [list(combinations(experiments_list, g)) for g in ranges]
    datasets = []
    for lista in tmp_set:
        for tupla in lista:
            datasets.append(list(tupla))
    return datasets


def matching_markers(df, markers_map):
    ndf = df.copy(deep=True)
    tmt = ndf.filter(regex='^TMT').columns.to_list()
    ndf.reset_index(inplace=True)
    ndf = ndf.loc[:, ~ndf.columns.isin(tmt)]
    merged = pd.merge(ndf, markers_map, on='Accession', how='left')
    merged.fillna(value={'marker': 'unknown'}, inplace=True)
    index_names = ndf.columns.to_list()
    indexes = zip(merged['Dataset'], merged['Accession'],
                  merged['calc.pI'], merged['Number.of.PSMs.per.Protein.Group'])
    merged.index = pd.MultiIndex.from_tuples(indexes, names=index_names)
    markers = merged['marker'].to_dict()
    return markers


def psms_sums_by_protgroup(sdf):
    df = sdf.copy(deep=True)
    mycols = df.filter(regex='PSMs.per.Protein.Group').columns.to_list()
    df['Sum.PSMs.per.Protein.Group'] = df.loc[:, mycols].sum(axis=1)
    df['quantiles'] = pd.qcut(df['Sum.PSMs.per.Protein.Group'
                              ].rank(method='first'),
                              q=[0, .25, .5, .75, 0.99, 1],
                              labels=[0, 0.25, 0.50, 0.75, 1],
                              duplicates='drop')
    df['PSMs_by_prot_group'] = pd.cut(df['Sum.PSMs.per.Protein.Group'],
                                      bins=6)
    return df


def progressive_df(df1, df2, outname, how, verbosity=False,
                   anot=pd.DataFrame()):
    fpath = os.path.join(os.getcwd(), f'{outname}.tsv')
    merged = pd.merge(df1, df2, left_index=True, right_index=True, how=how)
    merged_sorted = tmt_sorted_df(merged, lopit_utils.tmt_chans)
    if not anot.empty:
        anot.set_index('Accession', inplace=True)
        new_merged = pd.merge(merged_sorted, anot, left_index=True,
                              right_index=True, how=how)
        if verbosity:
            new_merged.to_csv(fpath, sep='\t', index=True)
    else:
        if verbosity:
            merged_sorted.to_csv(fpath, sep='\t', index=True)
    return merged


def accessory_data(entry1, entry2, entry3, global_df):
    if entry1 is not None:
        markers_map = pd.read_csv(entry1, sep=r'\t|,', header=0,
                                  engine='python')
        cnames = markers_map.columns.to_list()
        if ('Accession' or 'marker') not in cnames:
            print('marker file does not contain a column identified '
                  'as Accession or marker, or both')
        markers_map.marker.replace(' |-', '_', regex=True, inplace=True)
        markers_label = check_label_len(markers_map, 'marker')
        if len(markers_label) > 50:
            print('Markers infile contains over 50 unique entries '
                  '(markers + unknown).\nColor pallete for over 50 entries is '
                  'not currently supported in lopit_utils (line 28).\n'
                  'Exiting program...')
            sys.exit(-1)
    else:
        print('Markers file has not been declared. A mock list will be created')
        markers_map = pd.DataFrame()
    #   ---  reading input files or organizing dfs ---   #
    if entry2 is not None:
        features_df = pd.read_csv(entry2, sep="\t", header=0,
                                  engine='python')
    else:
        print('No protein feature file has been provided')
        features_df = pd.DataFrame()
    #   ---  reading input files or organizing dfs ---   #
    if entry3 is not None:
        additional_df = pd.read_csv(entry3, sep='\t', header=0,
                                    engine='python')
        if 'Accession' not in additional_df.columns.to_list():
            print('The additional file does not contain a column identified '
                  'as: Accession')
            additional_df = pd.DataFrame()
    else:
        print('No additional protein file has been provided')
        additional_df = pd.DataFrame()

    out_dfs = []
    for df in [(markers_map, 'marker'),
               (features_df, 'features file'),
               (additional_df, 'accessory file')]:
        dataframe, dtype = df
        if dataframe.empty:
            out_dfs.append(dataframe)
        else:
            # print(f'working on: {dtype}')
            out_dfs.append(lopit_utils.accesion_checkup(global_df,
                                                        dataframe,
                                                        dtype))

    nmarkers_map, nfeatures_df, nadditional_df = out_dfs
    return nmarkers_map, nfeatures_df, nadditional_df


def get_clusters(dfs_dic, dataset, markers_map, tsne_method, perplexity,
                 hdbscan_on_umap, features_df, epsilon, min_dist, min_size,
                 min_sample, n_neighbors, annotations_df, pca,
                 feature_projection, projections_enabled, verbosity):

    print(f'***   Working on Dataset {dataset}   ***')
    newdir = gr.new_dir(f'{dataset}')
    os.chdir(newdir)
    df_slice = dfs_dic[dataset]
    # ensure all TMT columns are the same type to avoid issues when transforming
    # with cudf
    dtypes = {col:'float64' for col in
              df_slice.filter(regex='^TMT').columns.to_list()}
    df_slice = df_slice.astype(dtypes)

    if markers_map.empty:  # mock dict
        markers = dict(zip(df_slice.index.values,
                           [1 for i in range(0, df_slice.shape[0])]))
    else:
        markers = matching_markers(df_slice, markers_map)
    if pca:
        #   ---   PCA framework   ---   #
        pca_99 = my_pca(df_slice, dataset, 0.99)
    else:
        pca_99 = None
    #   ---   tSNE framework   ---   #
    print('Generating tSNE embeddings')
    if perplexity is None:
        perplexity = np.floor(np.sqrt(df_slice.shape[0]))

    tsne_2coordinates = my_tsne(df=df_slice,
                                dataset=dataset,
                                marker=markers,
                                components=2,
                                method=tsne_method,
                                perplex=perplexity)
    # progressive dataframe pca + tsne 2 components
    if pca:
        pca_tsne_df_2 = progressive_df(df1=pca_99,
                                       df2=tsne_2coordinates,
                                       outname='PCA_tSNE2c',
                                       how='outer',
                                       verbosity=verbosity)
    else:
        pca_tsne_df_2 = tsne_2coordinates

    if sklearn:
        tsne_3coordinates = my_tsne(df=df_slice,
                                    dataset=dataset,
                                    marker=markers,
                                    components=3,
                                    method=tsne_method,
                                    perplex=perplexity)
        del tsne_3coordinates['marker']
        # progressive dataframe pca + tsne 3 components
        pca_tsne_df_3 = progressive_df(df1=pca_tsne_df_2,
                                       df2=tsne_3coordinates,
                                       outname='PCA_tSNE3c',
                                       how='outer',
                                       verbosity=verbosity)
    else:
        pca_tsne_df_3 = pd.DataFrame()

    print('Generating UMAP embeddings')
    #   ---   UMAP framework   ---   #
    umap_coordinates = my_umap(dfs=(df_slice, pca_99),
                               dataset=dataset,
                               marker=markers,
                               min_dist=min_dist,
                               n_neighbors=n_neighbors,
                               verbosity=verbosity)
    # progressive dataframe pca + tsne + umap
    if pca:
        oname = 'PCA_tSNE_UMAP'
    else:
        oname = 'tSNE_UMAP'
    if not pca_tsne_df_3.empty:
        pca_tsne_umap_df = progressive_df(df1=pca_tsne_df_3,
                                          df2=umap_coordinates,
                                          outname=oname,
                                          how='inner',
                                          verbosity=verbosity)
    else:
        pca_tsne_umap_df = progressive_df(df1=pca_tsne_df_2,
                                          df2=umap_coordinates,
                                          outname=oname,
                                          how='inner',
                                          verbosity=verbosity)

    print('Generating HDBSCAN clusters')
    # ---   HDBSCAN framework on TMT expression   ---   #
    hdbs_e, stats_e, persistence_e, sum_e = hdbscan_workflow(df=df_slice,
                                                             dataset=dataset,
                                                             dftype='TMT',
                                                             offset=3,
                                                             epsilon=epsilon,
                                                             min_size=min_size,
                                                             min_sample=min_sample)
    # progressive dataframe pca + tsne + umap + hdbscan_tmt
    progress_df1 = progressive_df(df1=pca_tsne_umap_df,
                                  df2=hdbs_e,
                                  outname=f'Coordinates_TMT_{dataset}',
                                  how='inner')
    # HDBSCAN framework on UMAP.d1 and UMAP.d2
    umap_cols = progress_df1.filter(regex=r'^UMAP.+2c$').columns.to_list()
    umap_coords = progress_df1.loc[:, umap_cols]
    hdbs_u, stats_u, persistence_u, sum_e = hdbscan_workflow(df=umap_coords,
                                                             dataset=dataset,
                                                             dftype='UMAP',
                                                             offset=3,
                                                             epsilon=epsilon,
                                                             min_size=min_size,
                                                             min_sample=min_sample)
    # progressive dataframe pca + tsne + umap + hdbscan_tmt + hdbscan_umap
    progress_df2a = progressive_df(df1=progress_df1,
                                  df2=hdbs_u,
                                  outname=f'Coordinates_UMAP_{dataset}',
                                  how='inner',
                                  anot=annotations_df)

    #  --- mapping markers onto hdbscan predictions ---   #
    if 'marker' in progress_df2a.columns.to_list():
        progress_df2b = predominant_marker_mmapper(df=progress_df2a,
                                    hdbscan_colname='hdb_labels_euclidean_TMT')
        progress_df3 = predominant_marker_mmapper(df=progress_df2b,
                                    hdbscan_colname='hdb_labels_manhattan_TMT')
    else:
        progress_df3 = progress_df2a.copy(deep=True)

    #   --- df by user defined group with tmt and coordinates   ---   #
    progress_df4 = progressive_df(df1=df_slice,
                                  df2=progress_df3,
                                  outname=f'Coordinates_ALL_{dataset}',
                                  how='inner',
                                  verbosity=True)

    if projections_enabled:
        print('Creating projections')
        #   --- HDBSCAN cluster projections  --  #
        if markers_map.empty:
            myloop = create_myloop(perplexity=perplexity,
                                   add_umap=hdbscan_on_umap,
                                   pca=pca,
                                   marker_info='')
        else:
            myloop = create_myloop(perplexity=perplexity,
                                   add_umap=hdbscan_on_umap,
                                   pca=pca,
                                   marker_info='marker',
                                   pred=True)

        projections_on_coordinates = projections(odf=progress_df3,
                                                 dataset=dataset,
                                                 sizes=(10, 50),
                                                 loop_dic=myloop)

        newdir = gr.new_dir(f'TMT_abundance_by_cluster')
        os.chdir(newdir)
        for key in myloop.keys():
            print(f'working on {key}')
            a = gr.dist_abundance_profile_by_cluster(df=progress_df4,
                                                     labels=key[0],
                                                     dataset=dataset)
        os.chdir('..')

    # #   ---  df integration with protein features  ---  #
    if not features_df.empty:
        print('Integration of protein features to main dataframe')
        integrated_df = df_integration(df1=progress_df4,
                                       df2=features_df,
                                       dataset=dataset)
        if feature_projection:
            print('Creating feature projections')
            #   ---   Feature projections   ---   #
            feat_tsne = feature_projections(df=integrated_df,
                                            coord1=f'tSNE_dim_1_2c_{perplexity}',
                                            coord2=f'tSNE_dim_2_2c_{perplexity}',
                                            dataset=dataset,
                                            c_type='tSNE',
                                            dic=features_loop)
            feat_umap = feature_projections(df=integrated_df,
                                            coord1='UMAP_dim1_2c',
                                            coord2='UMAP_dim2_2c',
                                            dataset=dataset,
                                            c_type='UMAP',
                                            dic=features_loop)
        #
    else:
        integrated_df = progress_df4.copy(deep=True)
        print('Protein features dataframe is empty:\n'
              '1) Ignore this message if you did not provide the file on '
              'purpose, otherwise something is wrong with the provided file\n'
              '2) If feature_projection flag was specified it will be \n'
              'overridden as there is not feature df file to be integrated \n'
              'with main dataframe')

    # #   ---  df integration with annotations  ---  #
    if not annotations_df.empty:
        print('Adding accessory information provided')
        final_merged = pd.merge(integrated_df,
                                annotations_df,
                                on='Accession',
                                how='left')
        fpath = os.path.join(os.getcwd(),
                             f'Final_df_{dataset}.AccessoryInfo.tsv')
        final_merged.to_csv(fpath, sep='\t', index=False)
    else:
        print(f"If you did not provide 'protein feature' or 'accessory files',\n\
              your FINAL FILE IS:', f'Coordinates_ALL_{dataset}_df.tsv")

    os.chdir('../..')
    # *-*-* garbage collection *-*-* #
    collected = gc.collect()
    print(f'{collected} garbage objects were collected')
    return


def cluster_analysis(files_list, features, datasets, tsne_method,
                     perplexity, mymarkers, fileout, hdbscan_on_umap,
                     epsilon, mindist, min_size, min_sample, n_neighbors,
                     additional_info, pca, feature_projection,
                     projections_enabled, verbosity):

    print('*** - Beginning of clustering workflow - ***\n')

    my_dfs = create_dic_df(files_list)
    newdir = lopit_utils.create_dir('Step5__Clustering',
                                    f'{fileout}')
    os.chdir(newdir)
    #   ---   datasets combinations   ---   #
    if datasets == 'all':
        datasets = create_combinations(list(my_dfs.keys()))
        dfs_dic = dataset_grouping(my_dfs, datasets, 'all')

    else:
        dfs_dic = dataset_grouping(my_dfs, datasets, 'user_defined')

    datasets = list(dfs_dic.keys())
    print('Requested datasets are:\n', datasets)

    #   ---  accession checkup and update of additional files  ---   #
    global_df = pd.DataFrame(pd.concat([my_dfs[k].loc[:, 'Accession']
                                            for k in my_dfs.keys()]))
    global_df.drop_duplicates(inplace=True)
    markers_map, features_df, annotations_df = accessory_data(mymarkers,
                                                              features,
                                                              additional_info,
                                                              global_df)
    #  clustering each dataset in parallel   #
    all_clst = Parallel(n_jobs=-1)(delayed(get_clusters)
                                          (dfs_dic, dataset, markers_map,
                                           tsne_method, perplexity,
                                           hdbscan_on_umap, features_df,
                                           epsilon, mindist, min_size,
                                           min_sample, n_neighbors,
                                           annotations_df, pca,
                                           feature_projection,
                                           projections_enabled,
                                           verbosity)
                                   for dataset in datasets)
    #  calculate shared clusters in files from all dfs_paths
    #shared_clusters = lopit_utils.clusters_across_datasets(all_clst)

    #   ---
    print('Clustering workflow has finished. Bye ;)')
    return 'Done'


#   ---   Execute   ---   #
if __name__ == '__main__':
    single_dfs = glob.glob(
        "D:\\PycharmProjects\\LOPIT\\Frames_ready_for_clustering\\test*.tsv")
    # os.path.isfile()
    markers_file = sys.argv[1]  # 'markers.tsv'
    features = pd.read_csv(sys.argv[2], sep='\t', header=0)
    # sys.argv[2] is a preprocessed charm file:
    # "D:\PycharmProjects\Perkinsids\Charms\Charms_dataset.tsv"
    #
    aggregation = [['PL1'], ['PL1', 'PL2']]
    # aggregation = []
    method_tsne = 'barnes_hut'  # makeit user input
    # _ = cluster_analysis(single_dfs, features, aggregation,
                         # method_tsne, perplexity, markers_file, verbose)

