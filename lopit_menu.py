import os
import sys
import glob
import argparse
import pandas as pd
from textwrap import wrap

#  ---  top-level parser  ---  #
parser = argparse.ArgumentParser(prog='lopit_analyst')
subparsers = parser.add_subparsers(help='sub-command help',
                                   dest='subparser_name')

#  --- second level parsers ---   #
#   Definition of dictionaries for second level parsers   #
#   messages   #
psm_ms = {
    'm1': 'PSMs file',
    'm2': 'Corrected PSM file obtained during diagnostics step',
    'm3': 'Filtered PSM matrix',
    'm4': 'Filtered and mv removed PSM file',
    'm5': 'Protein information obtained by PD',
    'm6': 'Aggregated matrix by master protein',
}

unwrapped = {
    'm1': 'Required arguments',
    'm2': 'Description',
    'm3': 'Tab separated file containing: path to feature files and file type',
    'm4': 'File with complementing information for PSMs input file',
    'm5': 'declare either MinDet',
    'm5-1': 'channels to apply mnar method. if more than one, separate them '
            'with semicolons. e.g., PL1:TMT126,TMT131N;PL2:TMT127N'
            ' or PL1-PL2-PLN-PLO:TMT126,TMT131N',
    'm6': 'Declare either knn',
    'm6-1': 'channels to apply mnar method. if more than one, separate them '
          'with semicolons. e.g.,  PL1:TMT126,TMT131N;PL2:TMT127N'
          ' or PL1-PL2-PLN-PLO:TMT126,TMT131N',
    'm7': 'Declare whether Mascot or Sequest.HT was used to predict proteins '
          'in PD',
    'm8': 'Declare experiment to reconstitute. If there is more than one '
          'experiment use semicolon: e.g., PLO:PLO1,PLO2;PLN:PLN1,PLN2',
    'm9': 'Dash-separated combination of experiment groups to compare '
          'Use comma to specify multiple separate combinations'
          'them e.g., PLN-PLO;PLN-PLO-PL1 *** Note: if all combinations '
          'are needed then only type ALL',
    'm10a': 'File containing protein markers',
    'm10b': 'File containing annotations or any additional data to append to '
            'main matrix. It requires the field called: Accession for merging',
    'm11': 'Formatted file with protein features',
    'm12': 'Prefixes-spectrum file tuples (refer only to the BASE NAME shared '
           'by the files corresponding to that experiment . If more of one '
           'experiment, use semicolon to separate '
           'experiments: e.g., for experiments PL1 and PL2 each with several'
           'files: PL1_E1.raw,PL1_E2.raw... PL1_E7.raw and '
           'PL2_E1.raw,PL2_E2.raw... PL2_E7.raw then the declaration must be:'
           'PL1-PL1_E;PL2-PL2_E . similarly, if you declare File.ID fields',
    'm13': 'type either pheno_data, protein_features, psms, protein_data '
           'merge or markers_onto_coordinates or json_subset',
    'm14': 're-declare channels, using hyphen and separating by semicolons. '
           'e.g., to move TMT131 to TMT131N and TMT128C to TMT128 only in '
           'experiments PL1 and PL2: '
           'PL1:TMT131-TMT131N,TMT128C-TMT128;PL2:TMT131-TMT131N,'
           'TMT128C-TMT128',
    'm15t1': 'tSNE flag: barnes_hut or exact. if NVIDIA GPU available fft '
            'is also enabled. Default=exact',
    'm15t2': 'tSNE flag: perplexity value. It should be between 5-50, '
            'the closest to 50, the tighter the groups. '
             'Default=sqrt(dataset size)'
            ' root of dataset size',
    'm15h1': 'HDBSCAN flag: indicate the cluster_selection_epsilon, '
            'it should be between 0.0 < 0.05 otherwise very '
            'few clusters will be predicted. Default=0.0',
    'm15h2': 'HDBSCAN flag: select the smallest size grouping to be '
             'considered a cluster. Default=log(dataset size)',
    'm15h3': 'HDBSCAN flag: The larger the value of min_samples you provide, '
             'the more conservative the clustering, and clusters will be '
             'restricted to progressively more dense areas.'
             ' Default=log(dataset size) + 3',
    'm15u1': 'UMAP flag: indicate the minimum distance for umap clustering, '
            'the smaller it is the more compact the figure. Values should be '
            'between 0 - 1. Larger values decrease figure resolution .'
            'Default=0.1)',
    'm15u2': 'UMAP flag: indicate how to control the balance of local '
            'versus global structure in UMAP. Default=sqrt(dataset size)',
    'm16': 'provide the shared path of the files for clustering replacing the'
           'these will be collected by glob.glob  e.g. if files names are\n: '
           'df_for_clustering_1.tsv, df_for_clustering_2.tsv, '
           'df_for_clustering_3.tsv, then the correct input declaration will '
           'be: path\\to\\df_for_clustering',
    'm17': 'path to file containing markers linked to Accessions',
    'm18': 'path to file containing annotations, additional information '
           'linked to Accessions or tagm predictions',
    'm19': 'type either 2D or 3D for a two- or three-dimension figure',
    'm20': 'specify the column name to be considered as axis',
    'm21': 'column to be used for point sizes in the figure., e.g. '
           'hdb_outlier_scores_manhattan_TMT',
    'm22': 'hbscan clusters based onto umap coordinates will be drawn'
           ' (memory and time consuming task).  Default: False',
    'm23': 'path to file containing a json dump to be used to subset the'
           'input dataframe',
    'm24': 'column to use for coloring the figure., e.g. '
           'hdb_labels_euclidean_TMT',
    'm25': 'taxon name (as it appears in the PSM file) to exclude '
           'from the analysis eg. Human.',
    'm26': 'column to search the experiment prefixes in',
    'm27': 'signal/noise ratio threshold for filtering (integer). Default=10'
}

other_ms = {k: '\n'.join(wrap(unwrapped[k])) for k in unwrapped.keys()}

#  arguments by tasks
#   dic_0, dic_2, and dic_3 for:
#   feature preparation, data filtering, and missing value removal:
m = [other_ms['m3'], psm_ms['m2'], psm_ms['m3']]
dic_0, dic_2, dic_3 = ({'-i': ['--input', str, i, True],
                        '-o': ['--out_name', str, 'output prefix', True]}
                       for i in m)
#  0- data preparation
dic_0.update({'-t': ['--data-type', str, other_ms['m13'], True],
              '-e': ['--experiment-prefixes', str, other_ms['m12'], False],
              '-f': ['--use_column', str, other_ms['m26'], False],
              '-c': ['--rename-columns', str, other_ms['m14'], False],
              '-s': ['--search-engine', str, other_ms['m7'], False],
              '-m': ['--markers-file', str, other_ms['m17'], False],
              '-a': ['--additional-file', str, other_ms['m18'], False],
              '-d': ['--figure-dimension', str, other_ms['m19'], False],
              '-x': ['--x-axis', str, other_ms['m20'], False],
              '-y': ['--y-axis', str, other_ms['m20'], False],
              '-z': ['--z-axis', str, other_ms['m20'], False],
              '-w': ['--size', str, other_ms['m21'], False],
              '-r': ['--color', str, other_ms['m24'], False],
              '-p': ['--perplexity', str, other_ms['m15t2'], False],
              '-u': ['--hdbscan_on_umap', bool, other_ms['m22'], False],
              '-j': ['--json_dump', str, other_ms['m23'], False]})

#   1- diagnostics:
dic_1 = {'-i': ['--input', str, psm_ms['m1'], True],
         '-o': ['--out_name', str, 'output prefix', True],
         '-a': ['--accessory-data', str, other_ms['m4'], False]}

#   2 - filtering:
dic_2.update({'-e': ['--exclude_taxon', str, other_ms['m25'], False],
              '-sn': ['--signal_noise_threshold', int, other_ms['m27'],
                      False, 10]})

#   4- mv imputation and aggregation:
dic_4 = {'-i': ['--input', str,  psm_ms['m4'], True],
         '-a': ['--accessory_data', str, other_ms['m4'], True],
         '-p': ['--protein_data', str, psm_ms['m5'], True],
         '-m1': ['--mnar', str, other_ms['m5'], False],
         '-d': ['--channels_mnar', str, other_ms['m5-1'], False],
         '-m2': ['--mar', str, other_ms['m6'], True],
         '-k': ['--channels_mar', str, other_ms['m6-1'], True],
         '-r': ['--interlaced_reconstitution', str, other_ms['m8'], False],
         '-o': ['--out_name', str, 'output prefix', True]}

#   5- clustering:
dic_5 = {'-i': ['--input', str, other_ms['m16'], True],
         '-o': ['--out_name', str, 'output prefix', True],
         '-g': ['--group_combinations', str, other_ms['m9'], True, 'all'],
         '-m': ['--markers_file', str, other_ms['m10a'], False],
         '-f': ['--protein_features', str, other_ms['m11'], False],
         '-t': ['--method_tsne', str, other_ms['m15t1'], True, 'exact'],
         '-x': ['--perplexity', int, other_ms['m15t2'], False, 50],
         '-e': ['--cluster_selection_epsilon', float, other_ms['m15h1'],
                False, 0.0],
         '-cs': ['--min_size', int, other_ms['m15h2'], False],
         '-ms': ['--min_sample', int, other_ms['m15h3'], False],
         '-md': ['--min_dist', float, other_ms['m15u1'], False, 0.1],
         '-n': ['--n_neighbors', float, other_ms['m15u2'], False],
         '-u': ['--hdbscan_on_umap', bool, other_ms['m22'], False],
         '-a': ['--additional-file', str, other_ms['m10b'], False]}

#   6- full analysis post diagnostics:
dic_6 = dic_4.copy()
shared_dic = {k: dic_5[k] for k in dic_5.keys() if k not in ['-i', '-o']}
dic_6.update(shared_dic)

#  7- stand-alone machine learning classification
dic_7 = {'-i': ['--input', str, other_ms['m16'], True],
         '-o': ['--out_name', str, 'output prefix', True],
         '-m': ['--markers_file', str, other_ms['m10a'], False]}

#   --- subparsers  ---   #


def default(dic, k):
    if len(dic[k]) == 5:
        return dic[k][4]
    else:
        return None


submenus = ['data_prep', 'diagnostics', 'filtering', 'mv_removal',
            'imputation_aggregation', 'clustering', 'full_analysis',
            'ml_classification']
submenus = [subparsers.add_parser(i) for i in submenus]

groups = [submenu.add_argument_group(other_ms['m1'], other_ms['m2'])
          for submenu in submenus]
group0, group1, group2, group3, group4, group5, group6, group7 = groups
dics = [(dic_0, group0), (dic_1, group1), (dic_2, group2), (dic_3, group3),
        (dic_4, group4), (dic_5, group5), (dic_6, group6), (dic_7, group7)]

for item in dics:
    dic, group = item
    for key in dic:
        group.add_argument(key, dic[key][0],
                           type=dic[key][1],
                           help=dic[key][2],
                           required=dic[key][3],
                           default=default(dic, key))


#  --- argument checks and re-assignments  ---  #
def path_check(arg_in):
    if isinstance(arg_in, list):
        l = []
        for p in arg_in:
            if os.path.isfile(p):
                l.append(os.path.abspath(p))
        if isinstance(l, list) and len(l) > 1:
            l = [os.path.abspath(i) for i in l]
            return l
        if isinstance(l, list) and len(l) == 1:
            return os.path.abspath(l[0])
        if isinstance(l, str):
            l = [os.path.abspath(i) for i in l.split(' ')]
            return l
        else:
            print('Input argument is unreachable')
            sys.exit(-1)
    else:
        tmp = glob.glob(os.path.abspath(arg_in))
        if tmp and len(tmp) == 1:
            l = os.path.abspath(tmp[0])
            return l
        else:
            print(f'{arg_in} does not exist')
            return sys.exit(-1)


def argument_check(user_args):
    print('Checking arguments')
    input_files = ['input', 'accessory_data', 'protein_data', 'json_dump',
                   'markers_file', 'protein_features', 'additional_file']
    allowed_search_engines = ['Mascot', 'Sequest.HT']
    allowed_columns = ['Spectrum.File', 'File.ID']
    allowed_tsne_methods = ['exact', 'barnes_hut', 'fft']
    for k in user_args.keys():
        if k in input_files:
            if user_args[k] is None:
                pass
            else:
                if isinstance(user_args[k], pd.DataFrame):
                    pass
                if isinstance(user_args[k], list) and \
                        isinstance(user_args[k][0], pd.DataFrame):
                    pass
                else:
                    user_args.update({k: path_check(user_args[k])})
        if user_args[k] is not None and k == 'search_engine':
            if user_args[k] not in allowed_search_engines:
                print('search engine is not in allowed engines')
                return sys.exit(-1)
        if user_args[k] is not None and k == 'use_column':
            if user_args[k] not in allowed_columns:
                print('declared column is not in allowed columns')
                return sys.exit(-1)
        if k == 'tsne_method':
            if user_args[k] not in allowed_tsne_methods:
                print('unrecognized tsne methods, please choose among: '
                      'barnes_hut or exact')
                return sys.exit(-1)
        if k == 'perplexity':
            try:
                v = int(user_args[k])
                user_args.update({k: v})
            except:
                pass
        if k == 'hdbscan_on_umap':
            if user_args[k] is None:
                user_args.update({k: False})
    my_args = argument_processing(user_args)
    return my_args


def ren_cols(exps_prefs):
    newnames = {}
    for exp in exps_prefs.keys():
        for v in exps_prefs[exp]:
            v = v.split('-')
            if exp not in newnames:
                newnames[exp] = {v[0]: v[1]}
            else:
                newnames[exp].update({v[0]: v[1]})
    return newnames


def special_tuples(user_args, k):
    m = 'Entry does not seem to conform to input format:'
    if user_args[k] is not None:
        arg_val = break_by_semicolon(user_args, k)
        if isinstance(arg_val, dict):
            return arg_val
        else:
            if 'remainder' in arg_val:
                print(f'{m}, {k}, {user_args[k]} ')
                sys.exit(-1)
            else:
                return arg_val
    else:
        return user_args[k]


def break_by_semicolon(user_args, k):
    try:
        exps_prefs = {e.split(':')[0]: e.split(':')[1].split(',') for e
                      in user_args[k].split(';')}
    except:
        exps_prefs = user_args[k].split(';')
    return exps_prefs


def process_complex_tuples(user_args, k):
    m = 'Entry does not seem to conform to input format:'
    spec_treatment = ['channels_mnar', 'channels_mar',
                      'interlaced_reconstitution']
    if k in spec_treatment:
        return special_tuples(user_args, k, m)

    elif user_args[k] is not None:
        if ';' in user_args[k]:
            exps_prefs = break_by_semicolon(user_args, k)
            if k != 'rename_columns':
                try:
                    exp_tuples = {entry.split('-')[1]: entry.split('-')[0]
                                  for entry in exps_prefs}
                    return exp_tuples
                except:
                    return exps_prefs
            else:
                exp_tuples = ren_cols(exps_prefs)
                return exp_tuples
        else:
            if ',' in user_args[k]:
                print(f'failure to recognize argument {k}: {user_args[k]}')
                sys.exit(-1)
            if k == 'rename_columns':
                exp_ren = ren_cols(user_args[k])
                return exp_ren
            if k == 'figure_dimension':
                dim = user_args['figure_dimension'].upper()
                if dim not in ['2D', '3D']:
                    print(f'failure to recognize argument {k}: {user_args[k]}')
                    sys.exit(-1)
            else:
                try:
                    tup = user_args[k].split('-')
                    # tup = {tup[0]: tup[1]}
                    return tup
                except:
                    print(m, k, user_args[k])
                    sys.exit(-1)
    else:
        return user_args[k]


def group_combos(user_entry):
    if user_entry.lower() == 'all':
        return user_entry.lower()
    else:
        main_groups = user_entry.split(',')
        g = [g.split('-') for g in main_groups]
        return g


def argument_processing(user_args):
    spec_tuples = ['channels_mnar', 'channels_mar', 'interlaced_reconstitution']
    for k in user_args.keys():
        if k == 'experiment_prefixes' or k == 'rename_columns':
            ctup = process_complex_tuples(user_args, k)
            user_args[k] = ctup
        if k in spec_tuples:
            stup = special_tuples(user_args, k)
            user_args[k] = stup
        if k == 'group_combinations':
            user_args[k] = group_combos(user_args[k])
    return user_args


def checkargs(arguments):
    # print('checkargs', arguments)
    if arguments.subparser_name == 'data_prep':
        if arguments.data_type == 'psms':
            if arguments.experiment_prefixes is None:
                parser.error('you must specify the experiment prefixes')
        elif arguments.data_type == 'protein_data':
            if arguments.search_engine is None:
                parser.error('you must declare a search_engine flag: '
                             'typer either Mascot or Sequest.HT')
    if arguments.subparser_name == 'clustering':
        params = str(arguments.input).split('/')
        directory = '/'.join(params[:-1])
        suffix = params[-1]
        fdir = os.path.abspath(directory)
        ifile_path_full = os.path.join(fdir, suffix)
        input_files = glob.glob(f'{ifile_path_full}*')
        if input_files:
            print(f'files in consideration:\n', '\n'.join(input_files))
            arguments.input = input_files
        else:
            print(f'No file contains the declared suffix {arguments.input}')
            sys.exit(-1)

    return arguments


def arguments():
    args = parser.parse_args()
    _ = checkargs(args)
    dic_args = vars(args)
    clean_args = argument_check(dic_args)
    print('User arguments are:\n', clean_args)
    return clean_args


#  ---  Execute argument parsing  ---   #


if __name__ == '__main__':
    _ = arguments()

