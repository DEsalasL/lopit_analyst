import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from lopit_utils import colors as mycolors


def new_dir(experiment):
    new_dir = f'{experiment}'
    cwd = os.getcwd()
    dir = os.path.join(cwd, new_dir)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    else:
        print('Directory already exists')
    return dir


def plot_umap(embed, experiment, dimensions):
    newdir = new_dir(f'UMAP')
    os.chdir(newdir)
    loop_list = list(combinations(dimensions, 2))
    for coordinates in loop_list:
        x, y = coordinates
        vals = embed.marker.to_list()
        sns.scatterplot(x=x, y=y, data=embed, hue=vals,
                        palette=sns.color_palette(mycolors,
                                                  len(set(set(vals)))),
                        size='marker', sizes=(10, 10)
                        ).set(title=f'{experiment} UMAP projection')
        plt.savefig(f'{experiment}_{x}{y}.plot.pdf')
        plt.cla()
        plt.close()
    os.chdir('..')
    return 'Done'


def plot_tsne(df, dataset, components, perplex):

    newdir = new_dir('tSNE')
    os.chdir(newdir)
    if 'marker' not in df.columns.to_list():
        t = sns.scatterplot(x=f'tSNE_dim_1_2c_{perplex}',
                            y=f'tSNE_dim_2_2c_{perplex}', size='marker',
                            sizes=(2, 2), data=df)
        # t.set(title=f'{dataset} T-SNE projection')
    else:
        vals = df.marker.to_list()
        numb_colors = len(set(vals))  # number of compartments
        t = sns.scatterplot(x=f'tSNE_dim_1_2c_{perplex}',
                            y=f'tSNE_dim_2_2c_{perplex}', data=df,
                            hue='marker',
                            palette=sns.color_palette(mycolors,
                                                      numb_colors),
                            size=4, sizes=(4, 4))
    t.set(title=f'{dataset} T-SNE projection')
    sns.move_legend(t, loc='upper right', bbox_to_anchor=(1.18, 1.1),
                    fontsize=4)
    plt.savefig(f'{dataset}_t-SNE.plot_{components}c_{perplex}.pdf')
    plt.cla()
    plt.close()
    os.chdir('..')
    return 'Done'


def scree_plot(explained_variance_ratio, pc_list, dataset):
    perc_explained_variance_ratio = [i * 100 for i in explained_variance_ratio]
    scree_df = pd.DataFrame(zip(pc_list, perc_explained_variance_ratio),
                         columns=['PC', 'Variance explained (%)'])
    scree_plot = sns.barplot(x='PC', y='Variance explained (%)',
                             data=scree_df, color='c')
    scree_plot.set_title('Component variance')
    scree_plot.tick_params(labelsize=6)
    scp = scree_plot.get_figure()
    scp.savefig(f'{dataset}_screeplot.pdf')
    return 'Done'



def pca_plot(df, dataset):
    sns.scatterplot(data=df, x='PC1', y='PC2') # hue="Label")
    plt.title(f'{dataset} PCA', fontsize=16)
    plt.xlabel('PC1', fontsize=16)
    plt.ylabel('PC2', fontsize=16)
    plt.savefig(f'PCA_{dataset}.pdf')
    plt.cla()
    plt.close()
    return 'Done'


def correlation_matrix_plot(df, dataset, pc_list, weights):
    #  --  Generated correlation matrix plot for weights  --  #
    # do this: informative_weights = weights_df.loc[pc of interest]
    weights_df = pd.DataFrame.from_dict(dict(zip(pc_list, weights)))
    weights_df['variable'] = df.columns.values
    weights_df = weights_df.set_index('variable')
    ax = sns.heatmap(weights_df, annot=True, cmap='Spectral')
    plt.title(f'{dataset} correlation matrix', fontsize=16)
    plt.xlabel('Principal components', fontsize=16)
    plt.ylabel('TMT channel', fontsize=16)
    plt.savefig(f'correlation_matrix_{dataset}.pdf')
    plt.cla()
    plt.close()
    return 'Done'


def boxplot(cluster_persistence, dataset):
    sns.boxplot(data=cluster_persistence,
                x=cluster_persistence.experiment,
                y=cluster_persistence.cluster_persistence,
                hue=cluster_persistence.experiment, dodge=False)
    plt.savefig(f'Persistence_scores.by.{dataset}.plot.pdf')
    plt.cla()
    plt.close()
    return 'done'


def projection(data, x, y, hue, size, sizes, dataset, palette, title):
    plt.figure(figsize=(15, 10))
    sns.set_theme(style='white', font_scale=1)
    # sns.color_palette(palette=palette)#, as_cmap=True)
    clusters = sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size,
                               sizes=sizes, palette=palette)
    clusters.legend(title=title.split('_')[1])
    sns.move_legend(clusters, loc='upper right', bbox_to_anchor=(1.125, 1),
                    fontsize=6)
    plt.title(f'{dataset} clusters on {title} coords', fontsize=16)
    if '_' in x:
        x, y = ' '.join(x.split('_')), ' '.join(y.split('_'))
    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    title = title.replace(' ', '_')
    plt.savefig(f'{dataset}_hdbscan_clst_on_{title}_coords.pdf')
    plt.cla()
    plt.close()
    return 'Done'


# consider including my colors here instead of palette  #
def feature_projections(df, coord1, coord2, dataset, c_type, dic):
    for i in dic.keys():
        labels, probs, sizes, palette, outname = dic[i]
        pj = projection(df, coord1, coord2, labels, probs, sizes,
                          dataset, palette, f'{c_type}_{outname}')
    return 'Done'


def name_conversion(tmt_col_names):
    names = {}
    replacements1 = {'TMT126.1': 'TMT126091', 'TMT126.2': 'TMT126092',
                     'N.1': '096', 'N.2': '097', 'C.1': '099', 'C.2': '098'}
    replacements2 = {'TMT126': 'TMT1260', 'N': '098', 'C': '099'}
    for name in tmt_col_names:
        if '.' in name:
            pattern = '|'.join(replacements1.keys())
            names[name] = re.sub(pattern,
                                 lambda match: replacements1[match.group(0)],
                                 name)
        else:
            pattern = '|'.join(replacements2.keys())
            names[name] = re.sub(pattern,
                                 lambda match: replacements2[match.group(0)],
                                 name)
    return names


def mock_tag(data_list):
    v = 999
    tags = {}
    for col in data_list:
        v += 1
        tags[col] = f'TMT{v}'
    return tags


def pivot(df, labels):
    tmtcols = df.filter(regex='^TMT').columns.to_list()
    tags = mock_tag(tmtcols)
    inverted_tags = {int(tags[k][3:]): k for k in tags.keys()}
    sliced_df = df.copy(deep=True)
    sliced_df.rename(columns=tags, inplace=True)
    ldf = pd.wide_to_long(sliced_df, stubnames='TMT', j='Tag',
                          i=['Accession', 'Dataset', labels])
    ldf.reset_index(inplace=True)
    ldf['Tag'] = ldf['Tag'].map(inverted_tags)
    ldf.rename(columns={'TMT': 'Abundance'}, inplace=True)
    return ldf


def dist_abundance_profile_by_cluster(df, labels, dataset):
    if 'level_0' in df.columns.to_list():
        df.rename(columns={'level_0': 'level0'}, inplace=True)
    df.reset_index(inplace=True)
    new_df = pivot(df, labels)
    custom = {acc: '#0000CD' for acc in new_df.Accession.to_list()}

    plt.figure(figsize=(15, 10))
    wrap_at = int(np.ceil(len(set(df[labels].to_list()))/5))
    g = sns.relplot(
        data=new_df, x='Tag', y='Abundance', col=labels, col_wrap=wrap_at,
        hue='Accession', style=labels, kind='line', palette=custom)
    # sns.move_legend(g, loc='right', fontsize=6)
    sns.set_style('darkgrid',
                  {'grid.color': '.6', 'grid.linestyle': ':'})
    g.legend.remove()
    g.set_xticklabels(rotation=45, fontsize=6)
    plt.savefig(f'{dataset}.{labels}.Tags.plot.pdf')
    plt.cla()
    plt.close()
    return


if __name__ == '__main__':
    df_in = pd.read_csv(sys.argv[1], sep='\t', header=0, index_col=0)
    df_in.drop_duplicates(inplace=True)
