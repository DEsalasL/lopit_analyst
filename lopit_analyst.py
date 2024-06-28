''' #!/home/dsalas/.conda/envs/lopit_analyst/bin/python '''
import gc
import os
import sys
import charms
import lopit_utils
import pandas as pd
import lopit_menu as lm
import mv_removal as mvr
import data_filtering as flt
import psm_diagnostics as dia
import clustering_data as clt
import mv_imp_norm_aggr as iagg
import SVM_KNN_RF_clustering as sml


#   Info  #
__author__ = 'Dayana Salas-Leiva'
__email__ = 'ds2000@cam.ac.uk'
__version__ = '0.0.1'
#   End Info   #


def prepare_input(args):  # ---  workflow 0- subparser_name: feature_prep
    if args['data_type'] == 'psms':
        psms = lopit_utils.psm_matrix_prep(args['input'],
                                           args['out_name'],
                                           args['experiment_prefixes'],
                                           args['rename_columns'],
                                           args,
                                           args['use_column'])
        return psms
    elif args['data_type'] == 'protein_data':
        proteins = lopit_utils.proteins_prep(args['input'],
                                             args['out_name'],
                                             args['search_engine'],
                                             args)
        return proteins
    elif args['data_type'] == 'protein_features':
        features = charms.sequence_properties(args['input'],
                                              args['out_name'],
                                              args)
        return features
    elif args['data_type'] == 'pheno_data':
        pheno_data = lopit_utils.phenodata_prep(args['input'],
                                                args['out_name'],
                                                args)
        return pheno_data
    elif args['data_type'] == 'merge':
        if args['markers_file'] is None and args['additional_file'] is None:
            print('At least one file must be declared for merging')
            sys.exit(-1)
        if (args['markers_file'] is not None
                and args['additional_file'] is not None):
            merged_data = lopit_utils.df_merger(args['input'],
                                                args['markers_file'],
                                                args['additional_file'],
                                                args['out_name'])
        if args['markers_file'] is None and args['additional_file'] is not None:
            merged_data = lopit_utils.df_merger(args['input'],
                                                '',
                                                args['additional_file'],
                                                args['out_name'])
        return merged_data

    elif args['data_type'] == 'json_subset':
        subdf = lopit_utils.json_subset(args['input'], args['json'])
        return subdf
    elif args['data_type'] == 'markers_onto_coordinates':
        # make sure needed arguments are passed
        _ = lopit_utils.args_needed(args)
        mydir = lopit_utils.create_dir('Projections__',
                                       args['out_name'])
        os.chdir(mydir)

        if args['figure_dimension'].upper() == '3D':
            interact_3d = lopit_utils.figure_rendering(args['input'],
                                                       args['x_axis'],
                                                       args['y_axis'],
                                                       args['size'],
                                                       args['color'],
                                                       args['figure_dimension'],
                                                       args['z_axis'])
        elif args['figure_dimension'].upper() == '2D':
            marker_loop = clt.create_marker_loop(args['perplexity'])
            try:
                projections = clt.projections(args['input'], '',
                                              (10, 50), marker_loop)
                interact_2d = lopit_utils.figure_rendering(args['input'],
                                                           args['x_axis'],
                                                           args['y_axis'],
                                                           args['size'],
                                                           args['color'],
                                                    args['figure_dimension'])
            except:
                interact_2d = lopit_utils.figure_rendering(args['input'],
                                                           args['x_axis'],
                                                           args['y_axis'],
                                                           args['size'],
                                                           args['color'],
                                                    args['figure_dimension'])
        else:
            a = args['figure_dimension']
            print(f'Unrecognized argument {a}')
            sys.exit(-1)
    else:
        offender = args['data_type']
        print(f'Input argument -t {offender} for data_type is not allowed.')
        sys.exit(-1)


def diagnosis(args):  # workflow 1- subparser: diagnostics
    density = False
    write_out = True
    diagnostic = dia.run_diagnostics(args['input'],
                                     args['accessory_data'],
                                     density,
                                     write_out,
                                     args['out_name'])
    return diagnostic


def filter_raw_data(args):  # workflow 2- subparser: filtering
    density = False
    write_out = True
    first_filtered_df = flt.run_data_filter(args['input'],
                                            density,
                                            write_out,
                                            args['out_name'],
                                            args['exclude_taxon'],
                                            args['signal_noise_threshold'])
    return first_filtered_df


def mv_removal(args):  # workflow 3- subparser: mv_removal
    mv_removed = mvr.run_heatmap_explorer(args['input'], args['out_name'])
    return mv_removed


def impute_and_aggregate(args):  # workflow 4- subparser: imputation-aggregation
    if args['mar'] is None and args['mnar'] is None:
        print('channels for at least imputation method must be declared')
        sys.exit(-1)

    imp_aggregated = iagg.imp_agg_normalize(args['input'],
                                            args['accessory_data'],
                                            args['protein_data'],
                                            args['out_name'],
                                            args,
                                            args['mnar'],
                                            args['channels_mnar'],
                                            args['mar'],
                                            args['channels_mar'],
                                            args['interlaced_reconstitution'])
    return imp_aggregated


def cluster_data(args):  # workflow 5- subparser: clustering
    clusters = clt.cluster_analysis(args['input'],
                                    args['protein_features'],
                                    args['group_combinations'],
                                    args['method_tsne'],
                                    args['perplexity'],
                                    args['markers_file'],
                                    args['out_name'],
                                    args['hdbscan_on_umap'],
                                    args['cluster_selection_epsilon'],
                                    args['min_dist'],
                                    args['min_size'],
                                    args['min_sample'],
                                    args['n_neighbors'],
                                    args['additional_file'],
                                    args['balancing_method'])
    return clusters


def automated_analysis(args):  # workflow 6- subparser: full_analysis
    first_filter_df = filter_raw_data(args)

    # update input argument for mv removal
    args.update({'input': first_filter_df})
    mv_removed_df = mv_removal(args)

    # update input argument for imputation and aggregation

    args.update({'input': mv_removed_df})
    imp_agg_df = impute_and_aggregate(args)

    # update input argument for clustering

    args.update({'input': imp_agg_df})
    clusters = cluster_data(args)
    _ = gc.collect()
    return clusters


def cluster_classification(args):
    print('Supervised machine learning protocol has started')
    if (args['input'] is not None and
            os.path.isfile(os.path.abspath(args['input']))):
        idf = pd.read_csv(args['input'], sep='\t', header=0, engine='python')
    else:
        print('input provided does not Exist. Exiting program...')
        sys.exit(-1)
    fdf, marker_df = sml.supervised_clustering(idf,
                                               args['markers_file'],
                                               train_size=0.7,
                                               accuracy_threshold=0.90)
    final_df = sml.write_df([idf, fdf, marker_df],
                            args['out_name'])
    return final_df


#   ---   Execute program   ---   #

#  --- user arguments ---  #


arguments = lm.arguments()


#   ---   subroutines   ---   #
if arguments['subparser_name'] == 'data_prep':
    output_dir = lopit_utils.create_dir('Formatted_input_data',
                                        arguments['out_name'])
    if arguments['data_type'] != 'markers_on_coordinates':
        os.chdir(output_dir)
    _ = prepare_input(arguments)
elif arguments['subparser_name'] == 'diagnostics':
    _ = diagnosis(arguments)
elif arguments['subparser_name'] == 'filtering':
    _ = filter_raw_data(arguments)
elif arguments['subparser_name'] == 'mv_removal':
    _ = mv_removal(arguments)
elif arguments['subparser_name'] == 'imputation_aggregation':
    _ = impute_and_aggregate(arguments)
elif arguments['subparser_name'] == 'clustering':
    _ = cluster_data(arguments)
elif arguments['subparser_name'] == 'full_analysis':
    _ = automated_analysis(arguments)
elif arguments['subparser_name'] == 'ml_classification':
    _ = cluster_classification(arguments)
