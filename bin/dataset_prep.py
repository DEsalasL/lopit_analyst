import os
import sys
import lopit_utils
import pandas as pd
from functools import reduce
from itertools import combinations
from natsort import natsorted
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


def dataset_to_dic(dic):
    newdic = {}
    for experiment in dic.keys():
        df_copy = dic[experiment].copy(deep=True)
        df_copy['Dataset'] = experiment
        miindex_df = multi_indexing_df(df_copy, experiment)
        dtypes = {col: 'float64' for col in
                  miindex_df.filter(regex='^TMT').columns.to_list()}
        miindex_df = miindex_df.astype(dtypes)
        newdic[experiment] = miindex_df
    return newdic


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
    dtypes = {col: 'float64' for col in
              df.filter(regex='^TMT').columns.to_list()}
    df = df.astype(dtypes)
    return df


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


def check_label_len(df, label):
    vals = list(df[label].unique())
    if 'unknown' in vals:
        vals.remove('unknown')
    vals = natsorted(vals)
    return vals


def accessory_data(entry1, entry2, global_df):
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
            print('Markers file contains over 50 classes (unique compartments'
                  ' + unknown).\nColor palette for over 50 classes is not '
                  'currently supported in lopit_utils (lines 31-40).\n'
                  'Exiting program...')
            sys.exit(-1)
    else:
        print('Markers file has not been declared. A mock list will be created')
        markers_map = pd.DataFrame()
    #   --- reading input files or organizing dfs ---   #
    if entry2 is not None:
        features_df = pd.read_csv(entry2, sep="\t", header=0,
                                  engine='python')
    else:
        print('No protein feature file has been provided')
        features_df = pd.DataFrame()

    out_dfs = []
    for df in [(markers_map, 'marker'),
               (features_df, 'features file')]:
        dataframe, dtype = df
        if dataframe.empty:
            out_dfs.append(dataframe)
        else:
            # print(f'working on: {dtype}')
            out_dfs.append(lopit_utils.accesion_checkup(global_df,
                                                        dataframe,
                                                        dtype))

    nmarkers_map, nfeatures_df= out_dfs
    return nmarkers_map, nfeatures_df,


#----
def data_prep(files_list, mymarkers, features, datasets,
              fileout):

    print('*** - Preparing datasets - ***\n')

    my_dfs = create_dic_df(files_list)
    newdir = lopit_utils.create_dir('Step5__Clustering',
                                    f'{fileout}')
    # moving into the new directory
    os.chdir(newdir)
    #   --- datasets combinations ---   #
    if datasets == 'all':
        datasets = create_combinations(list(my_dfs.keys()))
        dfs_dic = dataset_grouping(my_dfs, datasets, 'all')

    else:
        dfs_dic = dataset_grouping(my_dfs, datasets, 'user_defined')

    datasets = list(dfs_dic.keys())
    print('Requested datasets are:\n', datasets)

    #   --- accession checkup and update of additional files  ---   #
    global_df = pd.DataFrame(pd.concat([my_dfs[k].loc[:, 'Accession']
                                        for k in my_dfs.keys()]))
    global_df.drop_duplicates(inplace=True)
    markers_map, features_df= accessory_data(mymarkers,
                                                              features,
                                                              global_df)
    return dfs_dic, markers_map, features_df
