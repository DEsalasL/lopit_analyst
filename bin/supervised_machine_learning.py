import gc
import os
import sys
import svm
import rf
import time
import lopit_utils
import numpy as np
import pandas as pd
import ensemble as en
from functools import reduce
from natsort import natsorted
from collections import Counter
import utilities_locpro as utils_pro
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif


def majority_rule(row, min_count):
    counter = Counter(row)
    most_common_value, count = counter.most_common(1)[0]
    return most_common_value if count >= min_count else 'unknown'


def model_predictions(best_model,
                      X_train, X_val, X_test, X_val_probs,
                      y_test, unseen_array, markers_rev,
                      model_name, dataset_name):

    preds_probs = predict_and_evaluate(best_model=best_model,
                                       X_train=X_train,
                                       X_val=X_val,
                                       X_val_probs=X_val_probs,
                                       X_test=X_test,
                                       y_test=y_test,
                                       model_name=model_name,
                                       dataset_name=dataset_name,
                                       markers_rev=markers_rev)

    # get predictions for unseen data
    pred_unseen = best_model.predict(unseen_array)
    prob_unseen = best_model.predict_proba(unseen_array)
    preds_probs.update({'unseen_pred': pred_unseen,
                        'unseen_prob': prob_unseen})
    return preds_probs


def predict_and_evaluate(best_model, X_train, X_val,
                         X_val_probs, X_test,
                         y_test, markers_rev,
                         model_name, dataset_name):

    # Predict and evaluate using test
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)

    print('Generating classification report ...')
    _ = utils_pro.my_report(y_true=y_test,
                            y_pred=y_test_pred,
                            y_prob=y_test_prob,
                            markers_map=markers_rev,
                            model_name=model_name,
                            dataset_name=dataset_name)

    # get predictions and evaluations for training
    y_train_pred = best_model.predict(X_train)
    y_train_prob = best_model.predict_proba(X_train)

    if X_val_probs is not None:
        # get predictions and evaluations for validation
        y_val_pred = best_model.predict(X_val)
        y_val_prob = X_val_probs     # unbiased probabilities
    else:
        y_val_pred = None
        y_val_prob = None

    # accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set size: {len(y_test)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")

    return {'y_test_pred': y_test_pred,
            'y_test_prob': y_test_prob,
            'y_train_pred':y_train_pred,
            'y_train_prob':y_train_prob,
            'y_val_pred': y_val_pred,
            'y_val_prob': y_val_prob,
            'test_accuracy': accuracy}


def reconstitute_dfs(X_train_prob, y_train_pred, train_indices,
                     X_val_prob, y_val_pred, val_indices,
                     X_test_prob, y_test_pred, test_indices,
                     unseen_indices, unseen_prob, unseen_pred,
                     markers_rev, original_markers,
                     model_name, threshold):

    if X_val_prob is None:
        suffix = 'uncalibrated'
    else:
        suffix = 'calibrated'

    # predictions df for training data
    df_train = pd.DataFrame(X_train_prob, index=train_indices)
    df_train[f'{model_name}.pred'] = y_train_pred
    df_train['used in'] = 'training'

    if X_val_prob is not None:
        # predictions df for validation data
        df_val = pd.DataFrame(X_val_prob, index=val_indices)
        df_val[f'{model_name}.pred'] = y_val_pred
        df_val['used in'] = 'calibration'
    else:
        df_val = pd.DataFrame()

    # predictions df for testing data
    df_test = pd.DataFrame(X_test_prob, index=test_indices)
    df_test[f'{model_name}.pred'] = y_test_pred
    df_test['used in'] = 'test'

    # predictions df for unseen data
    df_unseen = pd.DataFrame(unseen_prob, index=unseen_indices)
    df_unseen[f'{model_name}.pred'] = unseen_pred
    df_unseen['used in'] = 'prediction'

    # concatenate predictions

    cat = pd.concat([df_train, df_val, df_test, df_unseen])
    cat.rename(columns=markers_rev, inplace=True)

    exclude = [f'{model_name}.pred', 'used in']
    # get best prediction based on threshold
    cols = [col for col in cat.columns.to_list() if col not in exclude]
    cat[f'{model_name}.{threshold}_pred'] =  np.where(
        cat.loc[:, cols].max(axis=1) >= threshold,
        cat.loc[:, cols].idxmax(axis=1), 'unknown')
    # map encoded markers to original marker names
    cat[f'{model_name}.pred'] = cat[f'{model_name}.pred'].map(markers_rev)
    cat[f'{model_name}.pred_best_prob_{suffix}'] = cat.loc[:, cols].max(axis=1)
    cat.index.name = 'Accession'

    # add original markers for cross-check accession assignment
    if 'Accession' in original_markers.columns:
        original_markers.set_index('Accession', inplace=True)

    merged_df = pd.merge(cat, original_markers, left_index=True,
                         right_index=True,  how='left')
    merged_df['marker'] = merged_df['marker'].fillna(value='unknown')

    # save df with synthetic markers
    merged_df.to_csv(f'{model_name}_extended_data.with.SM.tsv',
                     sep='\t', index=True)
    #save df without synthetic markers
    final_cat = merged_df[~merged_df.index.str.startswith('SM')]
    final_cat.to_csv(f'{model_name}_extended_data.without.SM.tsv',
                     sep='\t', index=True)
    # return only predictions and probabilities
    if model_name == 'RF':
        return final_cat.loc[:, ['used in',
                                 f'{model_name}.{threshold}_pred',
                                 f'{model_name}.pred_best_prob_{suffix}']]
    else:
        return final_cat.loc[:, [f'{model_name}.{threshold}_pred',
                                 f'{model_name}.pred_best_prob_{suffix}']]


def ensemble_best_params(df):
    df.reset_index(inplace=True)
    dic ={}
    for i, group in df.groupby('Classifier', as_index=True):
        d = dict(zip(group['Parameter'], group['Value']))
        dic[i] = d
    return dic


'''def selected_features(X_train, X_val, X_test, y_train,
                      unseen, cont_cols, cat_cols, cont_to_keep,
                      select_features):
    # reconstitute df
    if cat_cols:
        all_columns = cont_cols + cat_cols
    else:
        all_columns = cont_cols

    x_df = pd.DataFrame(X_train, columns=all_columns)
    if X_val is not None:
        x_vdf = pd.DataFrame(X_val, columns=all_columns)
    x_tdf = pd.DataFrame(X_test, columns=all_columns)
    unseen = pd.DataFrame(unseen, columns=all_columns)

    # Ensure y_train is 1D array
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()

    # keep only continuous columns for selection and select features
    if cont_to_keep:
        cocols = [col for col in cont_cols if
                  not col in cont_to_keep]
    else:
        cocols = cont_cols

    # df with cols to select from
    x = x_df.loc[:, cocols]
    selector = SelectKBest(score_func=f_classif,
                           k=select_features)
    selector.fit(x, y_train)
    kept_cols = list(x.columns.values[selector.get_support()])
    print(f'Columns kept for feature selection: {kept_cols}')
    if cont_to_keep:
        kept_cols.extend(cont_to_keep)
    if cat_cols:
        kept_cols.extend(cat_cols)

    print(f'Columns kept for classification analyses: {kept_cols}')
    to_drop = [col for col in all_columns if col not in kept_cols]

    #  drop selected columns
    x_df.drop(labels=to_drop, axis=1, inplace=True)
    if X_val is not None:
        x_vdf.drop(labels=to_drop, axis=1, inplace=True)
    x_tdf.drop(labels=to_drop, axis=1, inplace=True)
    unseen.drop(labels=to_drop, axis=1, inplace=True)

    # update x_train, x_val, x_test
    x_train = x_df.to_numpy(copy=True)
    if X_val is not None:
        x_val = x_vdf.to_numpy(copy=True)
    x_test = x_tdf.to_numpy(copy=True)
    unseen = unseen.to_numpy(copy=True)

    args_2_update = {'X_train': x_train,
                    'X_val': x_val,
                    'X_test': x_test,
                    'unseen_data': unseen,
                    'kept_columns': kept_cols}
    return args_2_update'''


def selected_features(X_train, X_val, X_test, y_train,
                      unseen_array, unseen_df, cont_cols, cat_cols,
                      cont_to_keep, select_features):
    # reconstitute df
    all_columns = cont_cols + cat_cols
    dfs = {'x_df': pd.DataFrame(X_train, columns=all_columns),
           'x_tdf': pd.DataFrame(X_test, columns=all_columns),
           'unseen_array': pd.DataFrame(unseen_array, columns=all_columns),
           'unseen_df': unseen_df.copy(deep=True)}

    if X_val.size != 0:
        dfs.update({'x_vdf': pd.DataFrame(X_val, columns=all_columns)})

    # Ensure y_train is 1D array
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()

    # keep only continuous columns for selection and select features
    if cont_to_keep is not None:
        cocols = [col for col in cont_cols if
                  not col in cont_to_keep]
    else:
        cocols = cont_cols
   #
    # df with cols to select from
    x = dfs['x_df'].loc[:, cocols]

    selector = SelectKBest(score_func=f_classif,
                           k=select_features)
    selector.fit(x, y_train)
    kept_cols = list(x.columns.values[selector.get_support()])

    if cont_to_keep is not None:
        kept_cols.extend(cont_to_keep)
    if cat_cols is not None:
        kept_cols.extend(cat_cols)

    print(f'Columns kept for classification analyses: {kept_cols}')
    to_drop = [col for col in all_columns if col not in kept_cols]

    #  drop selected columns
    for df in dfs.values():
        df.drop(labels=to_drop, axis=1, inplace=True)

    # update x_train, x_val, x_test

    x = zip(['x_df', 'x_tdf', 'unseen_array', 'unseen_df'],
            ['x_train', 'x_test', 'unseen_array', 'unseen_df'])

    dica = {}
    for k, new_key in x:
        dica[new_key] = dfs[k].to_numpy(copy=True)

    if X_val.size != 0:
        dica['x_val'] = dfs['x_vdf'].to_numpy(copy=True)
    else:
        dica['x_val'] = np.array([])


    with open('feature_selected_columns.txt', 'w') as f:
        f.write('\n'.join(kept_cols))

    args_2_update = {'X_train': dica['x_train'],
                    'X_val': dica['x_val'],
                    'X_test': dica['x_test'],
                    'unseen_data': dica['unseen_array'],
                    'unseen_df': dica['unseen_df'],
                    'kept_columns': kept_cols}

    return args_2_update


def supervised_clustering(odf, markers_df,
                          threshold, accuracy_threshold, smote_type,
                          accessory_file, scaling, scaling_method,
                          feature_selection, catcols, cocols, cocols_to_keep,
                          outname, sampling_strategy, n_jobs,
                          calibration, training_size,
                          test_size, calibration_size, augment_calibration):

    # sml classification
    print('Beginning of supervised learning classification ...',
          sep=' ', end='\n', file=sys.stdout, flush=True)

    # create training, validation, and test sets
    # Prepare the input data
    print('Generating training, validation, and test sets ...')

    #
    # Define a working df
    odf.set_index('Accession', inplace=True)
    if catcols is not None:
        working_df = odf.loc[:, cocols + catcols].copy(deep=True)
    else:
        working_df = odf.loc[:, cocols].copy(deep=True)

    data_ready = utils_pro.Data_prep(df=working_df,
                                     target_column='marker',
                                     markers=markers_df,
                                     categorical_variables=catcols,
                                     continuous_variables=cocols,
                                     continuous_to_keep=cocols_to_keep,
                                     calibration=calibration,
                                     scaling=scaling,
                                     scaling_method=scaling_method,
                                     device='CPU',
                                     dataset_name=outname,
                                     feature_ingineering=[],
                                     feature_selection=feature_selection,
                                     debug=False,
                                     verbose=False,
                                     smote_type=smote_type,
                                     sampling_strategy=sampling_strategy,
                                     train_size=training_size,
                                     test_size=test_size,
                                     calibration_size=calibration_size,
                                     augment_calibration=augment_calibration)

    # pre-split and get synthetic markers
    prepared_data= data_ready.prepare()

    print(f'{'=' * 60}\nWARNING:\n'
          'Be aware that scaling would not be advisable if proteomics data is the only \n'
          'source of data input and it has been previously normalized. if Scaling is \n'
          'applied in such a case, the predictions and their probabilities \n'
          'might be unreliable due to information loss caused by double scaling \n'
          f'{'=' * 60}')

    if scaling: # requested scaling
        print("Status: Data is being scaled ... ")
        working_data = data_ready.scale_data()
        working_args =  {'X_train': working_data["X_train_smote_scaled"],
                         'y_train': working_data["y_train_smote"],
                         'train_indices': working_data["train_smote_indices"],
                         'X_val': working_data["X_val_scaled"],
                         'y_val': working_data["y_val_array"],
                         'val_indices': working_data["val_indices"],
                         'X_test': working_data["X_test_scaled"],
                         'y_test': working_data["y_test_array"],
                         'test_indices': working_data["test_indices"],
                         'unseen_data': working_data["for_prediction_scaled"],
                         'markers_rev': prepared_data["markers_rev"],
                         'continuous_columns': prepared_data["continuous_columns"],
                         'categorical_columns': prepared_data["categorical_columns"],
                         'continuous_to_keep': prepared_data["continuous_to_keep"],
                         'feature_selection': prepared_data["feature_selection"]}
    else:
        print('Status: Data will not be scaled')
        # unscaled data
        working_args = {'X_train': prepared_data["X_train_smote"],
                        'y_train': prepared_data["y_train_smote"],
                        'train_indices': prepared_data["train_smote_indices"],
                        'X_val': prepared_data["X_val_array"],
                        'y_val': prepared_data["y_val_array"],
                        'val_indices': prepared_data["val_indices"],
                        'X_test':prepared_data["X_test_array"],
                        'y_test': prepared_data["y_test_array"],
                        'test_indices': prepared_data["test_indices"],
                        'unseen_data': prepared_data["for_prediction_unscaled"],
                        'markers_rev': prepared_data["markers_rev"],
                        'continuous_columns': prepared_data["continuous_columns"],
                        'categorical_columns': prepared_data["categorical_columns"],
                        'continuous_to_keep': prepared_data["continuous_to_keep"],
                        'feature_selection': prepared_data["feature_selection"]}

    # if feature selection requested
    if feature_selection is not None:
        to_update_dic = selected_features(X_train=working_args["X_train"],
                                          X_val=working_args["X_val"],
                                          X_test=working_args["X_test"],
                                          y_train=working_args["y_train"].copy(),
                                          unseen_array=working_args["unseen_data"],
                                          unseen_df=prepared_data["for_prediction_unscaled_df"],
                                          cont_cols=working_args["continuous_columns"],
                                          cont_to_keep=working_args["continuous_to_keep"],
                                          cat_cols=working_args["categorical_columns"],
                                          select_features=working_args["feature_selection"])

        working_args.update(to_update_dic)

    filtered_args = {k: v for k, v in working_args.items()
                     if k in ["X_train", "y_train", "X_val", "y_val"]}
    argsadd = {'n_jobs': n_jobs, 'calibration': calibration}
    filtered_args.update(argsadd)

    #
    results = [odf, markers_df]
    best_params = []
    accuracies = {}
    for classifier in ['SVM', 'RF', 'STACK']:# -> stack gives the best results
        print(f'Training {classifier} classifier ...')
        try:
            os.mkdir(classifier)
        except:
            print(f'Directory {classifier} already exists')
            pass
        os.chdir(classifier)
        if classifier == 'RF':
            best_model, best_param, X_val_probs, s = rf.rf_hyperparams(**filtered_args)
            accuracies[f'{classifier}_train'] = s['accuracy_train']
        elif classifier == 'SVM':
            best_model, best_param, X_val_probs, s = svm.svm_hyperparams(**filtered_args)
            accuracies[f'{classifier}_train'] = s['accuracy_train']
        else: # ensemble of SVM, RF
            bparam = pd.concat(best_params) # best parameters by model
            best_param_4_ensemble = ensemble_best_params(df=bparam)
            best_model, best_param, X_val_probs, s = en.create_ensemble(**filtered_args,
                                                                        params=best_param_4_ensemble,
                                                                        metaclassifier='RF')
            accuracies[f'{classifier}_train'] = s['accuracy_train']
        best_params.append(best_param)

        print(f'Carrying out predictions for {classifier} ...')
        pred_prob = model_predictions(best_model=best_model,
                                      model_name=classifier,
                                      dataset_name=outname,
                                      X_train=working_args["X_train"],
                                      X_val=working_args["X_val"],
                                      X_val_probs=X_val_probs,
                                      X_test=working_args["X_test"], # for evaluation
                                      y_test=working_args["y_test"], # for evaluation
                                      unseen_array=working_args["unseen_data"],
                                      markers_rev=working_args["markers_rev"])

        print('Creating prediction dataframe ...')
        unseen_index = prepared_data["for_prediction_unscaled_df"].index.to_list()
        dfd =reconstitute_dfs(train_indices=working_args["train_indices"],
                              val_indices=working_args["val_indices"],
                              test_indices=working_args["test_indices"],
                              X_train_prob=pred_prob['y_train_prob'],
                              y_train_pred=pred_prob['y_train_pred'],
                              X_val_prob=pred_prob['y_val_prob'],
                              y_val_pred=pred_prob['y_val_pred'],
                              X_test_prob=pred_prob['y_test_prob'],
                              y_test_pred=pred_prob['y_test_pred'],
                              unseen_indices=unseen_index,
                              unseen_prob=pred_prob['unseen_prob'],
                              unseen_pred=pred_prob['unseen_pred'],
                              markers_rev=prepared_data['markers_rev'],
                              original_markers=markers_df,
                              model_name=classifier,
                              threshold=threshold)
        results.append(dfd)

        accuracies[f'{classifier}_test'] = pred_prob['test_accuracy']
        accuracies[f'{classifier}_gap_between_train_and_test'] = (accuracies[f'{classifier}_train'] -
                                                                  accuracies[f'{classifier}_test'])
        os.chdir('..')

    bparam = pd.concat(best_params)  # best parameters by model
    bparam.to_csv(f'{outname}_best_parameters.by.classifier.tsv',
                  sep='\t', index=True)
    #
    accuracies_df = pd.DataFrame.from_dict(accuracies, orient='index',
                                           columns=['accuracy'])
    accuracies_df.to_csv(f'{outname}_accuracies.by.classifier.tsv',
                         sep='\t', index=True)
    #
    final_df = reduce(lambda left, right: pd.merge(left, right,
                                                   left_index=True,
                                                   right_index=True,
                                                   how='left'), results)
    final_df['marker'] = final_df['marker'].fillna(value='unknown')

    # if svm and rf are both present (for debug):
    cols = [f'SVM.{threshold}_pred', f'RF.{threshold}_pred']
    if set(final_df.columns.to_list()).intersection(set(cols)):
        final_df['SVM.RF.common'] = final_df.loc[:, cols].apply(
                        lambda row: majority_rule(row, min_count=3), axis=1)

    # add the accessory file if the file is provided
    if accessory_file is not None:
        oodf = odf.copy(deep=True)
        acc_file = lopit_utils.accesion_checkup(oodf.reset_index(), accessory_file,
                                                ftype='accessory file')
        acc_file.set_index('Accession', inplace=True)
        merged = pd.merge(final_df, acc_file, how='left',
                          left_index=True, right_index=True)
        merged.index.name = 'Accession'
        merged.to_csv(f'Final.{outname}.tsv',
                    sep='\t', index=True)
    else:
        final_df.index.name = 'Accession'
        final_df.to_csv(f'Final.{outname}.tsv',
                    sep='\t', index=True)
    print('Finished')
    return None

def marker_checkup(entry):
    if entry is not None and os.path.isfile(entry):
        markers_map = pd.read_csv(entry, sep='\t', header=0)
        cnames = markers_map.columns.to_list()

        if 'Accession' not in cnames or 'marker' not in cnames:
            print("Marker file does not contain a column identified as "
                  "'Accession' or 'marker', or both. Exiting program ...",
                  sep=' ', end='\n', file=sys.stdout, flush=True)
            sys.exit(-1)
    else:
        print('Declared marker file does NOT exist. Exiting program...',
               sep=' ', end='\n', file=sys.stdout, flush=True)
        sys.exit(-1)

    # check marker counts
    markers_count = markers_map['marker'].value_counts().reset_index()
    markers_count.columns = ['marker', 'count']
    not_enough_markers = markers_count[markers_count['count'] < 10]
    if not not_enough_markers.empty:
        print("WARNING: At least 10 markers per class are required.\n"
              "Exiting the program.\n"
              f"lack of markers in:\n {not_enough_markers}\n")
        sys.exit(0)
    mmap = markers_format(markers_map)
    return mmap


def markers_format(markersmap):
    markersmap.marker.replace(' |-', '_', regex=True, inplace=True)
    vals = list(markersmap['marker'].unique())
    vals.remove('unknown')
    vals = natsorted(vals)
    if len(vals) > 50:
        print(f'Markers infile contains over 50 classes (markers + unknown).\n'
              f'Color pallete for over 50 entries'
              'is not currently supported in lopit_utils.\n'
              'Exiting program...',
               sep=' ', end='\n', file=sys.stdout, flush=True)
        sys.exit(-1)
    return markersmap


def prep(infile, fileout, markers_file, additional_file,
         cat_cols, cont_cols, cont_keep):

    df = pd.read_csv(infile, sep='\t', header=0)
    if 'Accession' not in df.columns:
        print('No column labeled as `Accession` found in the input file.\n'
              'Exiting program ... ')
        sys.exit(-1)

    # check markers
    markers_df = marker_checkup(markers_file)

    # get data features for classification
    wrong_cols = []
    cocols = pd.read_csv(cont_cols, header=None).loc[:, 0].to_list()
    extra = [col for col in cocols if col not in df.columns.to_list()]
    wrong_cols.extend(extra)

    if cont_keep is not None:
        force_to_keep = pd.read_csv(cont_keep, header=None).loc[:, 0].to_list()
        extra = [col for col in force_to_keep if col not in df.columns.to_list()]
        if extra:
            wrong_cols.extend(extra)
    else:
        force_to_keep = None

    if cat_cols is not None:
        catcols = pd.read_csv(cat_cols, header=None).loc[:, 0].to_list()
        extra = [col for col in catcols if col not in df.columns.to_list()]
        if extra:
            wrong_cols.extend(extra)
    else:
        catcols = None

    if len(wrong_cols) > 0:
        print('The following user declared columns are not present\n'
              f'in the input data:\n{'\n'.join(wrong_cols)}\n'
              f'Exiting program ...', sep='\n', file=sys.stdout, flush=True)
        sys.exit(-1)

    if additional_file is not None and os.path.isfile(additional_file):
        acc_file = pd.read_csv(additional_file, sep='\t', header=0)
    else:
        acc_file = None

    # create a new dir to host the prediction outputs
    cwd = os.getcwd()
    newdir = os.path.join(cwd, f'Step6__SML_predictions_{fileout}')
    if not os.path.isdir(newdir):
        os.makedirs(newdir)
    else:
        print('Directory exists')
    #  check up files as df and pre-treatment
    newdir = os.path.abspath(newdir)
    os.chdir(newdir)
    return {'df': df,
           'markers_df': markers_df,
           'continuous_columns': cocols,
           'categorical_columns': catcols,
           'continuous_to_keep': force_to_keep,
           'additional_file': acc_file}


def prediction(main_df, balance_method, scaling, scaling_method,
               threshold, markers_df, cont_cols, cat_cols, cont_keep,
               feature_selection, additional_file, outname, sampling_strategy,
               n_jobs, calibration, training_fraction, test_fraction,
               calibration_fraction, augment_calibration):
    print('Starting parallel prediction\n')

    start= time.time()
    formatted_start = time.strftime("%H:%M:%S", time.localtime(start))
    print(f"Start time: {formatted_start}")
    #  predict each dataset in parallel
    supervised_clustering(odf=main_df,
                          markers_df=markers_df,
                          accuracy_threshold=0.90,
                          smote_type=balance_method,
                          outname=f'{outname}_SML',
                          accessory_file=additional_file,
                          threshold=threshold,
                          scaling=scaling,
                          scaling_method=scaling_method,
                          cocols=cont_cols,
                          cocols_to_keep=cont_keep,
                          catcols=cat_cols,
                          feature_selection=feature_selection,
                          sampling_strategy=sampling_strategy,
                          n_jobs=n_jobs,
                          calibration=calibration,
                          training_size=training_fraction,
                          test_size=test_fraction,
                          calibration_size=calibration_fraction,
                          augment_calibration=augment_calibration)
    end = time.time()
    formatted_end = time.strftime("%H:%M:%S", time.localtime(end))
    print(f"End time: {formatted_end}")
    elapsed = end - start
    e = lopit_utils.format_elapsed_time(elapsed)
    print(f"Elapsed time: {e}")

    return 'Program has finished'



## -- --  ##
if __name__ == '__main__':

    import argparse

    #  ---  top-level parser  ---  #
    parser = argparse.ArgumentParser(prog='svm_knn_rf_clustering.py',
                                     description='debugger')

    #  --- second level parsers ---   #
    m = {'m1.1': 'Input file containing features for prediction (TMT, calc.pI, '
                 'categorical variables, etc)',
         'm1.2': 'Input file containing markers by Accession numbers',
         'm2': 'Output suffix',
         'm3': 'Target column name. Default: marker',
         'm4': 'Minimum probability for selecting candidate '
               'markers (value between 0.0 and 1.0). Default: 1.0',
         'm7': 'Dataset name. Default: test',
         'm8': 'Accuracy threshold for filtering (value between 0.0 and 1.0). '
               'Default: 0.9',
         'm10': 'Accessory file to be merged at the end of the workflow '
                '(e.g., tSNE dimensions). Default: None',
         "m13": "File containing the headers of continuous columns "
                "(one header per line).",
         "m14": "File containing the headers of categorical columns "
                "(one header per line)",
         'm16': 'Data scaling. Default: False',
         'm20': 'Number of features to carry out prediction. Default: None',
         'm21': 'Scaling method. Choose from: StandardScaler, '
                'MinMaxScaler, MaxAbsScaler, RobustScaler. Default: RobustScaler',
         'm22': 'File containing the continuous columns headers to keep if '
                '`select_features` is requested.'
                '(one header per line), categorical columns are always kept. Default: None',
         'm23': 'Data augmentation method. Choose from: Smote, BorderlineSmote or SmoteNC. '
                'If categorical columns are declared: Default: SmoteNC, '
                'otherwise Default: Smote.',
         'm24': 'Sampling strategy for data augmentation. Options: minority, '
                'not_minority, not_majority, all, auto, keep_ratio. Default: auto',
         'm25': 'Number of jobs to run in parallel (integer). Default: 1',
         'm26': 'Calibrate probabilities. Note: it should only be applied if '
                'training dataset is large. Default: False',
         'm27': "Proportion of samples to be used as test set (value between 0,1). "
                "Default: 0.25",
         'm28': "Proportion of samples to be used for calibration. If "
                "calibration is requested (value between 0,1). Default: None",
         'm29': "Proportion of samples for training the model (value between 0,1). "
                "Default: 0.75",
         'm30': "Apply data augmentation to calibration set. When set to true the\n"
                "sampling_strategy declared will be used. Default: False"
         }


    parser.add_argument('-i', '--input', type=str, help=m['m1.1'], required=True)
    parser.add_argument('-m', '--markers_file', type=str, help=m['m1.2'], required=True)
    parser.add_argument('-con', '--continuous_columns', type=str, help=m['m13'], default=None)
    parser.add_argument('-f', '--continuous_to_keep', type=str, help=m['m22'], default=None)
    parser.add_argument('-cat', '--categorical_columns', type=str, help=m['m14'], default=None)
    parser.add_argument('-acc', '--additional_file', type=str, help=m['m10'], required=False)
    parser.add_argument('-o', '--out_name', type=str, help=m['m2'], required=True)
    parser.add_argument('-d', '--dataset_name', type=str, help=m['m7'], default='test')
    parser.add_argument('-t', '--target_column', type=str, help=m['m3'], default='marker')
    parser.add_argument('-p1', '--threshold', type=float, help=m['m4'], default=1.0)
    parser.add_argument('-s', '--scaling', type=bool, help=m['m16'], default=False)
    parser.add_argument('-sf', '--feature_selection', type=int, help=m['m20'], default=None)
    parser.add_argument('-sc', '--scaling_method', type=str, help=m['m21'], default='RobustScaler')
    parser.add_argument('-sm', '--balancing_method', type=str, help=m['m23'], default='Smote')
    parser.add_argument('-se', '--sampling_strategy', type=str, help=m['m24'], default='auto')
    parser.add_argument('-n', '--n_jobs', type=int, help=m['m25'], default=1)
    parser.add_argument('-cm', '--calibration', type=bool, help=m['m26'], default=False)
    parser.add_argument('-ts', '--test_fraction', type=float, help=m['m27'], default=0.25)
    parser.add_argument('-tt', '--training_fraction', type=float, help=m['m29'], default=0.75)
    parser.add_argument('-cs', '--calibration_fraction', type=float, help=m['m28'], default=None)
    parser.add_argument('-acs', '--augment_calibration_set', type=bool, help=m['m30'], default=False)


    args = parser.parse_args()
    dic_args = vars(args)
    '''log = f'{args['out_name']}.log'
    sys.stdout = open(log, 'w')'''
    print(args, file=sys.stdout, flush=True)

    main_dic = prep(infile=args.input,
                    fileout=args.out_name,
                    markers_file=args.markers_file,
                    additional_file=args.additional_file,
                    cat_cols=args.categorical_columns,
                    cont_cols=args.continuous_columns,
                    cont_keep=args.continuous_to_keep)

    if args.calibration is True and args.calibration_fraction is None:
        print('Calibration is requested but no calibration fraction is declared.\n'
              'Exiting program ...',)
        sys.exit(-1)
    if args.calibration_fraction is not None:
        if args.training_fraction + args.calibration_fraction + args.test_fraction != 1:
            print('The sum of training, calibration and test fractions is not 1.\n'
                  'Make sure you change the default values for training or test sets.\n'
                  'Exiting program ...')
            sys.exit(-1)
        if args.calibration_fraction > args.training_fraction:
            print(f'Training fraction is {args.training_fraction},\n'
                  f'Calibration fraction is {args.calibration_fraction},\n'
                  f'test_fraction is {args.test_fraction}.\n.')
            print('The training fraction must be larger than calibration. '
                  'Exiting program ...')
            sys.exit(-1)

    predictions = prediction(main_df=main_dic['df'],
                             balance_method=args.balancing_method,
                             threshold=args.threshold,
                             scaling=args.scaling,
                             scaling_method=args.scaling_method,
                             markers_df=main_dic['markers_df'],
                             additional_file=main_dic['additional_file'],
                             cat_cols=main_dic['categorical_columns'],
                             cont_cols=main_dic['continuous_columns'],
                             cont_keep=main_dic['continuous_to_keep'],
                             feature_selection=args.feature_selection,
                             outname=args.out_name,
                             sampling_strategy=args.sampling_strategy,
                             n_jobs=args.n_jobs,
                             calibration=args.calibration,
                             training_fraction=args.training_fraction,
                             test_fraction=args.test_fraction,
                             calibration_fraction=args.calibration_fraction,
                             augment_calibration=args.augment_calibration_set)
