import os
import hdbscan
import numpy as np
import pandas as pd
from sklearn import metrics
import graphic_results as gr


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
    # os.chdir('..')
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
        min_sample = int(np.floor(np.log(ndf.shape[0])))
    if min_sample is None:
        min_sample = min_size + offset
    # hdbscan clustering using mostly defaults.

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

