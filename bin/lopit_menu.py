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
    "m1": "PSMs input file",
    "m2": "Corrected PSM file obtained during diagnostics Step1",
    "m3": "Filtered PSM matrix obtained in Step2",
    "m4": "Processed PSM file obtained in Step3",
    "m5": "Formated protein file generated by PD (e.g., '*Proteins.txt')",
    "m6": "Aggregated matrix by master protein",
}

unwrapped = {
    "m1": "Arguments",
    "m2": "Description",
    "m3": "Tab separated file. if task is 'protein_feature' then this input "
          "contains the absolute paths to each feature file and the "
          "specification of the file type",
    "m4": "File with complementing information for PSMs input file. When "
          "using submenu 'imputation_aggregation', it corresponds to the "
          "formated phenotypic file",
    "m5": "Declare method MinDet (other methods will be added in the future)",
    "m5-1": "Declare channels for applying mnar method. Format: Experiment and"
            " channel must be separated with colons, each experiment "
            "declaration is separated by semicolons. e.g., 'PL1:TMT126,TMT131N"
            ";PL2:TMT127N' or if the same channels are to be declared for all "
            "experiments: 'PL1-PL2-PLN-PLO:TMT126,TMT131N'",
    "m6": "Declare method knn (other methods will be added in the future)",
    "m6-1": "Declare channels for applying mar method. Format: Experiment and"
            " channel must be separated with colons, each experiment "
            "declaration is separated by semicolons. e.g., 'PL1:TMT126,TMT131N"
            ";PL2:TMT127N' or if the same channels are to be declared for all "
            "experiments: 'PL1-PL2-PLN-PLO:TMT126,TMT131N'",
    "m7": "Declare which search engine was used by PD to identify the proteins "
          "in the mass spectra. options: 'Mascot' or 'Sequest.HT'",
    "m8": "Declare experiment(s) to reconstitute using IRS (only for "
          "experiments using an interlaced design). A new name must be "
          "provided for the experiments to be reconstituted, and the "
          "experiments must be separated by a comma. Use semicolons to "
          "separate multiple experiments to be reconstituted. For example, to "
          "create the 'PLO' experiment by integrating PLO1 and PLO2, and PLN "
          "by integrating PLN1 and PLN2 the format is 'PLO:PLO1,PLO2;PLN:PLN1,"
          "PLN2'",
    "m9": "To declare experiment combinations, separate each experiment "
          "combination with a dash, and each combination with semicolons "
          "e.g., For two combinations, one including PLN and PLO, and the "
          "other one including PLN, PLO, and PL1 the string is "
          "'PLN-PLO;PLN-PLO-PL1' Note that the dash in input line will be "
          "stripped and combinations will be reported as PLNPLO and PLNPLOPL1. "
          "If you need all combinations type 'ALL', this will lead to an "
          "automatic estimation of all possible combinations.",
    "m10a": "File containing protein markers",
    "m10b": "File containing annotations or any additional data to append to "
            "main the dataset. It must contain the column: 'Accession'",
    "m11": "Formatted file with protein features",
    "m12": "Values in 'File.ID' column that are linked to the spectra file.  "
           "It refers to the substring that is common among each experiment. "
           "e.g., experiments PL1 and PL2 are identified as f1 and f2 in the "
           "PD output. For PL1, each fraction is identified as f1.1,f1.2, f1.3,"
           "etc and f2.1, f2.2, etc. So the common substring for PL1 is f1 and "
           "for PL2 is f2. Use semicolon to separate multiple experiments. "
           "e.g., PL1-f1;PL2-f2. Note that the experiment denotations in the "
           "File.ID column of the PD input are setup by the user in PD.",
    "m13": "Options: 'pheno_data', 'protein_features', 'psms', "
           "'protein_data', 'merge', 'markers_onto_coordinates' or "
           "'json_subset'",
    "m14": "If you have to rename channels (due to the use of 10 plexes "
           "or a combination of 10 with either 11, 16 or other plexes), "
           "indicate the experiment followed by colon, then the old and new "
           "name separated by hyphen. Each renamed pair within an experiment "
           "must be separated with a comma, and each experiment separated by "
           "semicolons. For example, to rename TMT131 to TMT131N and TMT128C "
           "to TMT128 only in experiments PL1 and PL2, declare: ‘PL1:TMT131-"
           "TMT131N,TMT128C-TMT128;PL2:TMT131-TMT131N,TMT128C-TMT128’",
    "m15t1": "tSNE specific flag: barnes_hut or exact. if NVIDIA GPU available "
             "fft is also enabled. Default=exact",
    "m15t2": "tSNE specific flag: perplexity value. It should be between 5-50, "
             "the closest to 50 the tighter the groups will be. "
             "Default=sqrt(dataset size) root of dataset size",
    "m15h1": "HDBSCAN specific flag: indicate the value for the "
             "'cluster_selection_epsilon' parameter,  it should be between 0.0 "
             "< 0.05 otherwise very few clusters will be predicted. Default=0.0",
    "m15h2": "HDBSCAN specific flag: select the smallest group size to "
             "be considered a cluster. Default=log(dataset size)",
    "m15h3": "HDBSCAN specific flag: value for 'min_sample' parameter. The "
             "larger the value is, the more conservative the clustering is "
             "and clusters will be restricted to progressively more dense "
             "areas. Default=log(dataset size) + 3",
    "m15u1": "UMAP specific flag: indicate the minimum distance for UMAP "
             "clustering, the smaller it is the more compact the figure. "
             "Values should be between 0 - 1. Larger values decrease figure "
             "resolution. Default=0.1",
    "m15u2": "UMAP specific flag: control the balance of local versus global "
             "structure in UMAP. Default=sqrt(dataset size)",
    "m16": "Common substring associated to the files obtained in previous step "
           "The substring is shared among the multiple experiment "
           "files. For example, for files df_for_clustering_PL1.tsv, "
           "df_for_clustering_PL2.tsv, df_for_clustering_PLN.tsv, "
           "df_for_clustering_PLO.tsv, the correct substring is: "
           "'df_for_clustering'. Please provide an absolute or relative path: "
           "e.g., '\\path\\to\\df_for_clustering'.",
    "m16-1": "Path pointing to the directory 'Step5__Clustering_' or its "
             "equivalent when using sml as stand alone program.",
    "m17": "Path to file containing markers linked to Accessions",
    "m18": "Path to file containing annotations, TAGM predictions or any "
           "additional information linked to Accessions.",
    "m19": "Type either 2D or 3D for a two- or three-dimension figure, "
           "respectively",
    "m20": "Specify the column header to be considered as axis",
    "m21": "Column header to be used for point sizes in the figure., e.g. "
           "'hdb_outlier_scores_manhattan_TMT'",
    "m22": "HDBSCAN clusters based onto UMAP coordinates will be drawn. "
           "Memory and time consuming task.  Default: False",
    "m23": "Path to file containing a json dump to be used to subset the "
           "input data",
    "m24": "Column header to use for coloring the figure., e.g. "
           "'hdb_labels_euclidean_TMT'",
    "m25": "Taxon handle name (as it appears in the PSM file) to exclude "
           "from the analysis e.g., Human.",
    "m26": "Column header to search the experiment prefixes in",
    "m27": "Signal/noise ratio threshold used for filtering (integer). "
           "Default=10",
    "m28": "Balancing method used to draw synthetic markers. Options: "
           "'borderline', 'over_under', 'smote' or 'unbalanced'. 'unbalanced' "
           "will not generate synthetic markers, hence, it will use the "
           "markers as originally provided",
    "m28-1": "If the marker file contains only the columns 'Accession' and "
            "'marker' all those markers will be used for supervised machine "
            "learning in all combinations. This corresponds to 'global' type."
            "If instead, the marker file contains 'Accession', and multiple "
            "column headers that match the number and name of the main "
            "directories in Step5, then the program will match each column "
            "header (a.k.a. combination) to its respective input file in the "
            "Step5 directory (or its analog when using foreign data). This "
            "corresponds to 'local' type. Note: directory names "
            "correspond to the experiment combinations. options: 'global', "
            "'local'",
    "m29": "Unambiguous substring to recognize the input file. For example, "
           "if the file of interest starts with 'Final_df' and there are no "
           "other files that start with that substring, then the substring is "
           "'Final_df*'",
    "m30": "Minimum threshold to remove channels with missing values (value "
           "from 0 - 1) e.g., 0.1 will remove all channels containing more "
           "than 10 percent missing values, but will keep those channels that "
           "have less than 10 percent of missing values.",
    "m31": "A PCA analysis will be carried out. Memory and time "
           "consuming task when applied to large datasets. "
           "Default: False",
    "m32": "Turn on verbosity (type True). Default: False",
    "m33": "Projections will be drawn if protein features file is provided",
    "m34": "Create all HDBSCAN projections. Default: False"}

other_ms = {k: '\n'.join(wrap(unwrapped[k], 70))
            for k in unwrapped.keys()}

#  arguments by tasks
#   dic_0, dic_2, and dic_3 for:
#   feature preparation, data filtering, and missing value removal:
m = [other_ms['m3'], psm_ms['m2'], psm_ms['m3']]
dic_0, dic_2, dic_3 = ({'-i': ['--input', str, i, True],
                        '-o': ['--out_name', str, 'output prefix or suffix',
                               True],
                        '-v': ['--verbose', bool, other_ms['m32'], False]}
                       for i in m)

#  0- data preparation
dic_0.update({'-t': ['--data_type', str, other_ms['m13'], True],
              '-e': ['--experiment_prefixes', str, other_ms['m12'], False],
              '-f': ['--use_column', str, other_ms['m26'], False],
              '-c': ['--rename_columns', str, other_ms['m14'], False],
              '-s': ['--search_engine', str, other_ms['m7'], False],
              '-m': ['--markers_file', str, other_ms['m17'], False],
              '-a': ['--additional_file', str, other_ms['m18'], False],
              '-d': ['--figure_dimension', str, other_ms['m19'], False],
              '-x': ['--x_axis', str, other_ms['m20'], False],
              '-y': ['--y_axis', str, other_ms['m20'], False],
              '-z': ['--z_axis', str, other_ms['m20'], False],
              '-w': ['--size', str, other_ms['m21'], False],
              '-r': ['--color', str, other_ms['m24'], False],
              '-p': ['--perplexity', str, other_ms['m15t2'], False],
              '-u': ['--hdbscan_on_umap', bool, other_ms['m22'], False],
              '-j': ['--json_dump', str, other_ms['m23'], False]})

#   1- diagnostics:
dic_1 = {'-i': ['--input', str, psm_ms['m1'], True],
         '-o': ['--out_name', str, 'output prefix or suffix', True],
         '-a': ['--accessory_data', str, other_ms['m4'], False],
         '-v': ['--verbose', bool, other_ms['m32'], False]}

#   2 - filtering:
dic_2.update({'-e': ['--exclude_taxon', str, other_ms['m25'], False],
              '-sn': ['--signal_noise_threshold', int, other_ms['m27'],
                      False, 10]})

#  3 - mv removal:
dic_3.update({'-rm': ['--remove_columns', float, other_ms['m30'], False]})

#   4- mv imputation and aggregation:
dic_4 = {'-i': ['--input', str,  psm_ms['m4'], True],
         '-a': ['--accessory_data', str, other_ms['m4'], True],
         '-p': ['--protein_data', str, psm_ms['m5'], True],
         '-m1': ['--mnar', str, other_ms['m5'], False],
         '-d': ['--channels_mnar', str, other_ms['m5-1'], False],
         '-m2': ['--mar', str, other_ms['m6'], True],
         '-k': ['--channels_mar', str, other_ms['m6-1'], True],
         '-r': ['--interlaced_reconstitution', str, other_ms['m8'], False],
         '-o': ['--out_name', str, 'output prefix or suffix', True],
         '-v': ['--verbose', bool, other_ms['m32'], False]}

#   5- clustering:
dic_5 = {'-i': ['--input', str, other_ms['m16'], True],
         '-o': ['--out_name', str, 'output prefix or suffix', True],
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
         '-a': ['--additional_file', str, other_ms['m10b'], False],
         '-p': ['--pca', bool, other_ms['m31'], True],
         '-fp': ['--feature_projection', str, other_ms['m33'], False],
         '-pe': ['--projections_enabled', str, other_ms['m34'], False],
         '-v': ['--verbose', bool, other_ms['m32'], False]}

# #   6- full analysis post diagnostics:
# dic_6 = dic_4.copy()
# shared_dic = {k: dic_5[k] for k in dic_5.keys() if k not in ['-i', '-o']}
# dic_6.update(shared_dic)

#  7- stand-alone machine learning classification
dic_7 = {'-i': ['--input', str, other_ms['m16-1'], True],
         '-o': ['--out_name', str, 'output prefix or suffix', True],
         '-m': ['--markers_file', str, other_ms['m10a'], True],
         '-t': ['--markers_type', str, other_ms['m28-1'], True],
         '-r': ['--recognition_motif', str, other_ms['m29'], True],
         '-b': ['--balancing_method', str, other_ms['m28'], True],
         '-a': ['--additional_file', str, other_ms['m10b'], False],
         '-v': ['--verbose', bool, other_ms['m32'], False]}

#   --- subparsers  ---   #


def default(dic, k):
    if len(dic[k]) == 5:
        return dic[k][4]
    else:
        return None


submenus = ['data_prep', 'diagnostics', 'filtering', 'mv_removal',
            'imputation_aggregation', 'clustering', 'full_analysis',
            'sml']
submenus = [subparsers.add_parser(i) for i in submenus]

groups = [submenu.add_argument_group(other_ms['m1'], other_ms['m2'])
          for submenu in submenus]
group0, group1, group2, group3, group4, group5, group6, group7 = groups
dics = [(dic_0, group0), (dic_1, group1), (dic_2, group2), (dic_3, group3),
        (dic_4, group4), (dic_5, group5), (dic_7, group7)]
#eliminated (dic_6, group6) from dics above

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
    allowed_balancing_methods = ['borderline', 'over_under',
                                 'smote', 'unbalanced']
    allowed_markers_type = ['global', 'local']
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
        if k == 'balancing_method':
            if user_args[k] not in allowed_balancing_methods:
                print('unrecognized balancing method, please choose among: '
                      f'{allowed_balancing_methods}')
                return sys.exit(-1)
        if k == 'markers_type':
            if user_args[k] not in allowed_markers_type:
                print('unrecognized marker type, please choose among: '
                      f'{allowed_markers_type}')
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
        if k == 'pca':
            if user_args[k] is None:
                user_args.update({k: False})
        if k == 'verbose':
            if user_args[k] is None:
                user_args.update({k: False})
        if user_args[k] is not None and k == 'remove_columns':
            v = float(user_args[k])
            user_args.update({k: v})
        if user_args[k] is None and k == 'remove_columns':
            user_args.update({k: False})
        if user_args[k] is None and k == 'feature_projection':
            user_args.update({k: False})
        if user_args[k] is None and k == 'projections_enabled':
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

