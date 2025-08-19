import os
import sys
import lopit_utils
import numpy as np
import pandas as pd
import graphic_results as gr
import dataset_prep as dp
from matplotlib.colors import hex2color
from natsort import index_natsorted

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


def create_myloop(perplexity, add_umap, pca, marker_info='', pred=False):
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


def custom_color_dic(df, label):
    vals = dp.check_label_len(df, label)
    if len(vals) <= 50:
        custom_hex_colors = lopit_utils.colors[: len(vals)]
        custom_rgb_colors = [hex2color(color) for color in custom_hex_colors]
        color_dic = dict(zip(vals, custom_rgb_colors))
        color_dic.update({'unknown': hex2color('#cab2d6')})
        return color_dic
    else:
        print('Number of marker classes surpasses the number of colors in the '
              'palette (see lopit_utils lines 31-40)')
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

            os.chdir('../..')
    return 'Done'


def feature_projections(df, coord1, coord2, dataset, c_type, dic):
    cols = df.columns.to_list()
    updated_dic = {k: dic[k] for k in dic.keys() if dic[k][0] in cols}
    for i in updated_dic.keys():
        labels, probs, sizes, palette, outname = updated_dic[i]
        os.chdir(c_type)
        pj = gr.projection(df, coord1, coord2, labels, probs, sizes,
                           dataset, palette, f'{c_type}_{outname}')
        os.chdir('../..')
    return 'Done'

def create_projections(markers_map, progress_df3, progress_df4,
                       dataset, perplexity, hdbscan_on_umap, pca):
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
                                    pred=False)

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