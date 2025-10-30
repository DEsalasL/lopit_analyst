'''#!/home/dsalas/.conda/envs/lopit_analyst_2025/bin/python'''
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import charms
import lopit_utils
import lopit_menu as lm
import mv_removal as mvr
import data_filtering as flt
import psm_diagnostics as dia
import clustering_data as clt
import mv_imp_norm_aggr as iagg
import supervised_machine_learning as sml

#   Info  #
__author__ = ['Dayana Salas-Leiva']
__email__ = 'ds2000@cam.ac.uk'
__version__ = '1.0.'
#   End Info   #


def prepare_input(args):  # ---  workflow 0- subparser_name: feature_prep
    if args['data_type'] == 'psms':
        psms = lopit_utils.psm_matrix_prep(args['input'],
                                           args['out_name'],
                                           args['experiment_prefixes'],
                                           args['rename_columns'],
                                           args,
                                           args['use_column'],
                                           args['verbose'])
        return psms
    elif args['data_type'] == 'protein_data':
        proteins = lopit_utils.proteins_prep(args['input'],
                                             args['out_name'],
                                             args['search_engine'],
                                             args,
                                             args['verbose'])
        return proteins
    elif args['data_type'] == 'protein_features':
        features = charms.sequence_properties(args['input'],
                                              args['out_name'],
                                              args,
                                              args['verbose'])
        return features
    elif args['data_type'] == 'pheno_data':
        pheno_data = lopit_utils.phenodata_prep(args['input'],
                                                args['out_name'],
                                                args,
                                                args['verbose'])
        return pheno_data
    elif args['data_type'] == 'merge':
        if args['markers_file'] is None and args['additional_file'] is None:
            print('At least one file must be declared for merging',
                  sep=' ', end='\n', file=sys.stdout, flush=True)
            sys.exit(-1)
        if (args['markers_file'] is not None
                and args['additional_file'] is not None):
            merged_data = lopit_utils.df_merger(args['input'],
                                                args['markers_file'],
                                                args['additional_file'],
                                                args['out_name'],
                                                args['verbose'])
        if args['markers_file'] is None and args['additional_file'] is not None:
            merged_data = lopit_utils.df_merger(args['input'],
                                                '',
                                                args['additional_file'],
                                                args['out_name'],
                                                args['verbose'])
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
                                                       args['z_axis'],
                                                       args['verbose'])
            return None
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
                                                           args['figure_dimension'],
                                                           args['verbose'])
            except:
                interact_2d = lopit_utils.figure_rendering(args['input'],
                                                           args['x_axis'],
                                                           args['y_axis'],
                                                           args['size'],
                                                           args['color'],
                                                           args['figure_dimension'],
                                                           args['verbose'])
                return None
        else:
            a = args['figure_dimension']
            print(f'Unrecognized argument {a}')
            sys.exit(-1)
    else:
        offender = args['data_type']
        print(f'Input argument -t {offender} for data_type is not allowed.',
              sep=' ', end='\n', file=sys.stdout, flush=True)
        sys.exit(-1)


def diagnosis(args):  # workflow 1- subparser: diagnostics
    density = False
    write_out = args['verbose']
    diagnostic = dia.run_diagnostics(args['input'],
                                     args['accessory_data'],
                                     density,
                                     write_out,
                                     args['out_name'])
    return diagnostic


def filter_raw_data(args):  # workflow 2- subparser: filtering
    density = False
    first_filtered_df = flt.run_data_filter(args['input'],
                                            density,
                                            args['verbose'],
                                            args['out_name'],
                                            args['exclude_taxon'],
                                            args['signal_noise_threshold'])
    return first_filtered_df


def mv_removal(args):  # workflow 3- subparser: mv_removal
    if args['remove_columns'] is not None:  # boolean will become 0.0
        try:
            col_rm_threshold = float(args['remove_columns'])
        except ValueError:
            print(f"rm value is not a number: {args['remove_columns']}",
                  sep=' ', end='\n', file=sys.stdout, flush=True)
            sys.exit(-1)
        mv_removed = mvr.run_heatmap_explorer(args['input'], args['out_name'],
                                              col_rm_threshold, args['verbose'])
    else:
        mv_removed = mvr.run_heatmap_explorer(args['input'], args['out_name'],
                                              args['verbose'])
    return mv_removed


def impute_and_aggregate(args):  # workflow 4- subparser: imputation-aggregation
    if args['mar'] is None and args['mnar'] is None:
        print('channels for at least imputation method must be declared',
               sep=' ', end='\n', file=sys.stdout, flush=True)
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
                                            args['interlaced_reconstitution'],
                                            args['verbose'])
    return imp_aggregated


def cluster_data(args):  # workflow 5- subparser: clustering
    clusters = clt.cluster_analysis(files_list=args['input'],
                                    features=args['protein_features'],
                                    datasets=args['group_combinations'],
                                    tsne_method=args['tsne_method'],
                                    tsne_perplexity=args['tsne_perplexity'],
                                    tsne_learning_rate=args['tsne_learning_rate'],
                                    tsne_n_iter=args['tsne_n_iter'],
                                    tsne_init=args['tsne_init'],
                                    mymarkers=args['markers_file'],
                                    fileout=args['out_name'],
                                    hdbscan_epsilon=args['hdbscan_epsilon'],
                                    hdbscan_min_size=args['hdbscan_min_size'],
                                    hdbscan_min_sample=args['hdbscan_min_sample'],
                                    umap_n_neighbors=args['umap_n_neighbors'],
                                    umap_min_dist=args['umap_min_dist'],
                                    hdbscan_on_umap=args['hdbscan_on_umap'])
    return clusters


def predict_compartments(args):  # workflow 7 subparser sml

    log = f'{args['out_name']}.log'
    sys.stdout = open(log, 'w')
    print(args, file=sys.stdout, flush=True)

    main_dic = sml.prep(infile=args['input'],
                                fileout=args['out_name'],
                                markers_file=args['markers_file'],
                                additional_file=args['additional_file'],
                                cat_cols=args['categorical_columns'],
                                cont_cols=args['continuous_columns'],
                                cont_keep=args['continuous_to_keep'])

    predictions = sml.prediction(main_df=main_dic['df'],
                                 balance_method=args['balancing_method'],
                                 threshold=args['threshold'],
                                 scaling=args['scaling'],
                                 scaling_method=args['scaling_method'],
                                 markers_df=main_dic['markers_df'],
                                 additional_file=main_dic['additional_file'],
                                 cat_cols=main_dic['categorical_columns'],
                                 cont_cols=main_dic['continuous_columns'],
                                 cont_keep=main_dic['continuous_to_keep'],
                                 feature_selection=args['feature_selection'],
                                 outname=args['out_name'],
                                 sampling_strategy=args['sampling_strategy'],
                                 n_jobs=args['n_jobs'],
                                 calibration=args['calibration'],
                                 augment_calibration=args['augment_calibration_set'],
                                 training_fraction=args['training_fraction'],
                                 test_fraction=args['test_fraction'],
                                 calibration_fraction = args['calibration_fraction'])
    return predictions


#   ---   Execute program   ---   #

def main():
    #  --- user arguments ---  #

    arguments = lm.arguments()
    '''
    log = f'lopit_analyst_{arguments['subparser_name']}.log'
    sys.stdout = open(log, 'w')'''

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
    elif arguments['subparser_name'] == 'sml':
        _ = predict_compartments(arguments)


if __name__ == '__main__':
    main()