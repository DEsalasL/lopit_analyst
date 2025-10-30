import sys
import os
import glob
import gc
from math import log
import pandas as pd
from typing import Dict, Any, List
import dataset_prep as dp
import gpu_cpu_pca as gc_pca
import gpu_cpu_tsne as gc_tsne
import gpu_cpu_umap as gc_umap
import cpu_hdbscan
from collections import Counter
import gpu_memory_management as gmm
import graphic_results as gr
from functools import reduce

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
        if index[0] == 'unknown':
            z = 'unknown'
        else:
            z = predominant_marker(g)
        tmp_dic[index[0]] = z
    df[f'{hdbscan_colname}_pred_marker'] = df[hdbscan_colname].map(tmp_dic)
    return df


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


def process_single_dataset(df: pd.DataFrame, dataset_name: str,
                           markers_map: pd.DataFrame,
                           pca_comp: float = 0.99,
                           tsne_method: str = 'barnes_hut',
                           tsne_perplexity: float = 50.0,
                           tsne_learning_rate: int = 350,
                           tsne_n_iter: int = 10000,
                           tsne_init: str = 'random',
                           umap_neighbors: int = 15,
                           umap_min_dist: float = 0.1,
                           hdbscan_epsilon: float = 0.025,
                           hdbscan_min_size: int = 6,
                           hdbscan_min_sample: int = 9,
                           hdbscan_on_umap: bool = False)->Dict[str, Any]:
    """Process a single dataset with PCA, t-SNE, UMAP and HDBSCAN"""

    results = {
        'dataset': dataset_name,
        'pca_result': None,
        'tsne_result': None,
        'umap_result': None,
        'hdbscan_result': None,
        'success': {'pca': False, 'tsne': False,
                    'umap': False, 'hdbscan': False}
    }

    print(f"\n=== Processing dataset: {dataset_name} ===")
    print(f"Dataset shape: {df.shape}")

    newdir = gr.new_dir(dataset_name)
    os.chdir(newdir)

    # Sequential processing to avoid GPU memory conflicts
    # 1. PCA
    newdir = gr.new_dir('PCA')
    os.chdir(newdir)
    pca_result = gc_pca.gpu_pca_safe(df=df,
                                     dataset=dataset_name,
                                     comp=pca_comp)
    if pca_result is not None:
        results['pca_result'] = pca_result
        results['success']['pca'] = True
        print(f"✓ PCA completed: {pca_result.shape}")
        # Save result
        pca_result.to_csv(f'PCA_results_{dataset_name}.tsv', sep='\t',
                          index=True)
        os.chdir('..')
    else:
        print(f'Something went wrong with PCA for dataset {dataset_name}.')
        sys.exit(1)

    # 2. t-SNE
    newdir = gr.new_dir('tSNE')
    os.chdir(newdir)
    tsne_result = gc_tsne.gpu_tsne_safe(df=df,
                                        dataset=dataset_name,
                                        method=tsne_method,
                                        perplexity=tsne_perplexity,
                                        learning_rate=tsne_learning_rate,
                                        n_iter=tsne_n_iter,
                                        init=tsne_init)

    if tsne_result is not None:
        results['tsne_result'] = tsne_result
        results['success']['tsne'] = True
        print(f"✓ t-SNE completed: {tsne_result.shape}")
        # Save result
        tsne_result.to_csv(f'tSNE_results_{dataset_name}.tsv', sep='\t',
                           index=True)
        os.chdir('..')
    else:
        print(f'Something went wrong with t-SNE for dataset {dataset_name}.')
        sys.exit(1)

    # 3. UMAP
    newdir = gr.new_dir('UMAP')
    os.chdir(newdir)
    umap_result = gc_umap.gpu_umap_safe(dfs=(df, pca_result),
                                        dataset=dataset_name,
                                        n_neighbors=umap_neighbors,
                                        min_dist=umap_min_dist,
                                        n_components=2)
    if umap_result is not None:
        results['umap_result'] = umap_result
        results['success']['umap'] = True
        print(f"✓ UMAP completed: {umap_result.shape}")
        # Save result
        umap_result.to_csv(f'UMAP_results_{dataset_name}.tsv', sep='\t',
                           index=True)
        os.chdir('..')
    else:
        print(f'Something went wrong with UMAP for dataset {dataset_name}.')
        sys.exit(1)

    # 4. HDBSCAN
    print(f"Processing HDBSCAN on CPU")
    newdir = gr.new_dir('HDBSCAN')
    os.chdir(newdir)
    if hdbscan_min_size is None:
       hdbscan_min_sample = int(log(df.shape[0]) + 3)
    else:
        hdbscan_min_sample = int(hdbscan_min_size)
    print('Generating HDBSCAN clusters')
    # ---   HDBSCAN framework on TMT expression -> euclidean and manhattan --- #
    hdbs_e, stats_e, persistence_e, sum_e = cpu_hdbscan.hdbscan_workflow(
                                            df=df,
                                            dataset=dataset_name,
                                            dftype='TMT',
                                            offset=3,
                                            epsilon=hdbscan_epsilon,
                                            min_size=hdbscan_min_size,
                                            min_sample=hdbscan_min_sample,
                                            )

    # HDBSCAN framework on UMAP.d1 and UMAP.d2
    umap_cols = umap_result.filter(regex=r'^UMAP.+2c$').columns.to_list()
    umap_coords = umap_result.loc[:, umap_cols]
    hdbs_u, stats_u, persistence_u, sum_e = cpu_hdbscan.hdbscan_workflow(
                                         df=umap_coords,
                                         dataset=dataset_name,
                                         dftype='UMAP',
                                         offset=3,
                                         epsilon=hdbscan_epsilon,
                                         min_size=hdbscan_min_size,
                                         min_sample=hdbscan_min_sample)

    hdbs_uu = None
    if hdbscan_on_umap: # needs to be
        # HDBSCAN framework on all UMAP coordinates
        regex = r'^UMAP_dim\d+_(?:[3-9]|[1-9]\d|100)c$'
        umap_cols = umap_result.filter(regex=regex).columns.to_list()

        print(umap_cols)
        umap_coords = umap_result.loc[:, umap_cols]
        hdbs_uu, stats_uu, persistence_uu, sum_eu = (
            cpu_hdbscan.hdbscan_workflow(
            df=umap_coords,
            dataset=dataset_name,
            dftype='UMAP_mult_coord',
            offset=3,
            epsilon=hdbscan_epsilon,
            min_size=hdbscan_min_size,
            min_sample=hdbscan_min_sample))
        if hdbs_uu is not None:
            results['hdbscan result on UMAP'] = hdbs_uu
            results['success']['hdbscan on UMAP'] = True
            print(f"✓ HDBSCAN completed: {hdbs_uu.shape}")
            # Save result
            hdbs_uu.to_csv(f'HDBSCAN_CPU_results_'
                           f'{dataset_name}.UMAP_multiple_coordinates.tsv',
                           sep='\t', index=True)

    # collecting results info
    if hdbs_e is not None:
        results['hdbscan_result on TMT'] = hdbs_e
        results['success']['hdbscan on TMT'] = True
        print(f"✓ HDBSCAN completed: {hdbs_e.shape}")
        # Save result
        hdbs_e.to_csv(f'HDBSCAN_CPU_results_{dataset_name}.TMT.tsv', sep='\t',
                           index=True)
    if hdbs_u is not None:
        results['hdbscan_result on UMAP'] = hdbs_u
        results['success']['hdbscan on UMAP'] = True
        print(f"✓ HDBSCAN completed: {hdbs_u.shape}")
        # Save result
        hdbs_u.to_csv(f'HDBSCAN_CPU_results_{dataset_name}.UMAP_2-coord.tsv',
                      sep='\t', index=True)

        # finish and go back to previous dir
        os.chdir('..')

    else:
        print(f'Something went wrong with HDBSCAN for dataset {dataset_name}.')
        sys.exit(1)
    # 5. merge all dfs from steps 1-4
    base_lst = [df, pca_result, tsne_result, umap_result, hdbs_e, hdbs_u,
                hdbs_uu]
    list_of_dfs = [e for e in base_lst if e is not None]

    merged_dfs = reduce(lambda left, right: pd.merge(left, right,
                                                     left_index=True,
                                                     right_index=True,
                                                     how='left'),
                         list_of_dfs)

    merged_dfs.reset_index(inplace=True)
    res_markers = f'All_results_{dataset_name}.markers_mapped.tsv'
    if not markers_map.empty:
        merged_markers = pd.merge(merged_dfs, markers_map,
                                  on='Accession',
                                  how='left')
        nm_dfs1 = predominant_marker_mmapper(
            merged_markers,
            hdbscan_colname='hdb_labels_euclidean_TMT')
        nm_dfs2 = predominant_marker_mmapper(
            nm_dfs1,
            hdbscan_colname='hdb_labels_manhattan_TMT')
        if hdbscan_on_umap:
            nm_dfs3 = predominant_marker_mmapper(
                      nm_dfs2,
                      hdbscan_colname='hdb_labels_euclidean_UMAP_mult_coord')
            nm_dfs4 = predominant_marker_mmapper(
                      nm_dfs3,
                      hdbscan_colname='hdb_labels_manhattan_UMAP_mult_coord')
            nm_dfs4.to_csv(res_markers, sep='\t', index=False)
        else:
            nm_dfs2.to_csv(res_markers, sep='\t', index=False)
    else:
        res_no_markers = f'All_results_{dataset_name}.no_markers_mapped.tsv'
        merged_dfs.to_csv(res_no_markers, sep='\t', index=False)
    os.chdir('..')

    return results


def process_all_datasets(datasets: List[pd.DataFrame], dataset_names: List[str],
                        **kwargs) -> List[Dict[str, Any]]:
    """Sequential processing"""

    # Initialize RMM once at the start
    gpu_initialized = gmm.initialize_rmm_once()
    if not gpu_initialized:
        print("GPU initialization failed. Exiting program")
        sys.exit(1)

    all_results = []

    for i, (df, name) in enumerate(zip(datasets, dataset_names)):
        print(f"\n{'=' * 60}")
        print(f"Processing dataset {i + 1}/{len(datasets)}: {name}")
        print(f"{'=' * 60}")

        try:
            result = process_single_dataset(df, name, **kwargs)
            all_results.append(result)

            # Force cleanup between datasets
            gmm.cleanup_gpu_memory()

        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            # Continue with next dataset
            gmm.cleanup_gpu_memory()
            continue

    return all_results


def cluster_analysis(files_list,
                     features,
                     datasets,
                     tsne_method,
                     tsne_perplexity,
                     tsne_learning_rate,
                     tsne_n_iter,
                     tsne_init,
                     mymarkers,
                     fileout,
                     hdbscan_epsilon,
                     hdbscan_min_size,
                     hdbscan_min_sample,
                     umap_n_neighbors,
                     umap_min_dist,
                     hdbscan_on_umap):

    print('*** - Beginning of clustering workflow - ***\n')
    # prepare datasets and move to a new working directory
    x = dp.data_prep(files_list=files_list,
                     mymarkers=mymarkers,
                     features=features,
                     datasets=datasets,
                     fileout=fileout)

    dfs_dic, markers_map, features_df = x

    # patch for unpacking datasets
    dsets_names = []
    dsets_vals = []
    for key in dfs_dic.keys():
        dsets_names.append(key)
        dsets_vals.append(dfs_dic[key])

    # Process all datasets
    results = process_all_datasets(
        datasets=dsets_vals,
        dataset_names=dsets_names,
        pca_comp=0.99,
        tsne_method=tsne_method,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_n_iter=tsne_n_iter,
        tsne_init=tsne_init,
        umap_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        markers_map=markers_map,
        hdbscan_epsilon=hdbscan_epsilon,
        hdbscan_min_size=hdbscan_min_size,
        hdbscan_min_sample=hdbscan_min_sample,
        hdbscan_on_umap=hdbscan_on_umap)

    # Analyze results
    successful_pca = sum(1 for r in results if r['success']['pca'])
    successful_tsne = sum(1 for r in results if r['success']['tsne'])
    successful_umap = sum(1 for r in results if r['success']['umap'])

    print(f"\n{'=' * 60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total datasets processed: {len(results)}")
    print(f"Successful PCA: {successful_pca}/{len(results)}")
    print(f"Successful t-SNE: {successful_tsne}/{len(results)}")
    print(f"Successful UMAP: {successful_umap}/{len(results)}")

    return results


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

