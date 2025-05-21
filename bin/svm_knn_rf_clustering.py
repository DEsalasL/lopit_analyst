import glob
import os
import re
import sys
import warnings
import lopit_utils
import numpy as np
import pandas as pd
from sklearn import svm
from functools import reduce
from natsort import natsorted
from collections import Counter
from joblib import Parallel, delayed
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline as mp
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, train_test_split)
from sklearn.preprocessing import (StandardScaler,  LabelEncoder)
from sklearn.metrics import (classification_report, multilabel_confusion_matrix,
                             accuracy_score, precision_recall_curve, roc_curve,
                             precision_score, recall_score, f1_score)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR


sml_cols = ['SVM.prediction', 'KNN.prediction', 'Random.forest.prediction',
            'Naive.Bayes', 'most.common.pred.SVM.KNN.RF.NB.hdbscan',
            'most.common.pred.SVM.KNN.RF.NB', 'most.common.pred.supported.by',
            'best.pred.supported3+', 'best.pred.supported4+']


def predef_message(ml_method, acc, req_acc, dataset):
    m = (f'******** WARNING  {ml_method} ********\n'
         f'Accuracy for {dataset} is {acc}, which is BELOW the requested '
         f'accuracy threshold of {req_acc}\n'
         '*************************\n')
    print(m)
    return 'Done'


def markers_map(marker_df):
    markers = list(marker_df['marker'].unique())
    classes = [i for i in range(0, len(markers))]
    dic = dict(zip(markers, classes))
    inv_dic = {dic[k]: k for k in dic.keys()}
    return dic, inv_dic


def borderline_smote_estimation(x, y, neighbors=2):
    # # transform the dataset to balance the sampling classes
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-clas
    # sification/
    oversample = BorderlineSMOTE(k_neighbors=neighbors)
    x, y = oversample.fit_resample(x, y)
    return x, y


def over_under_smote(x, y, neighbors=2):
    # transform the dataset to balance the sampling clasess
    # define pipeline using 3 points (k_neighbors: 3-1), oversample
    # the minority class at sampling_over (i.e., 0.3 -> 30%) and
    # undersample the majority class at sampling_under (i.e., 0.5 -> 50%)
    over = SMOTE(k_neighbors=neighbors,
                 sampling_strategy='not majority', random_state=123)
    under = RandomUnderSampler(sampling_strategy='majority')
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    x, y = pipeline.fit_resample(x, y)
    return x, y


def print_labels(all_markers_df, y_train, y_test):
    print('*** Warning train and test markers are: ')
    marker_labels = sorted(Counter(all_markers_df['marker']).items(),
                           key=lambda x: x[1], reverse=True)
    train_labels = sorted(y_train.value_counts().items(),
                          key=lambda x: x[1], reverse=True)
    test_labels = sorted(y_test.value_counts().items(),
                         key=lambda x: x[1], reverse=True)
    for info in [f'all markers: {len(marker_labels)}\n {marker_labels}',
                 f'train markers: {len(train_labels)}\n {train_labels}',
                 f'test markers: {len(test_labels)}\n {test_labels}']:
        print(info)
    return 'Done'


def check_neighbors_and_sample(n_neighbors, markers_list, markers_dic, exp):
    ml = markers_list.tolist()
    m = {item: ml.count(item) for item in set(ml)}
    offender = {}
    for i in m.keys():
        if m[i] <= n_neighbors:
            offender[i] = m[i]
    if offender:
        translated = {markers_dic[k]: offender[k] for k in offender.keys()}
        print(f'Not enough markers for smote {exp}:\n{translated}\n'
              f'Exiting program')
        sys.exit(1)


def create_sets(x, y, train_size, markers_acc, markers_dic, exp, btype):
    # create training and test dataset for balanced classes

    X_train, X_temp, Y_train, Y_temp = train_test_split(x,
                                                        y,
                                                        test_size=train_size,
                                                        random_state=42,
                                                        stratify=y)

    print(f'Total makers used for training: {len(Y_train)}')
    # get original and synthetic markers for training
    train_markers = original_n_synthetic_markers(df=markers_acc,
                                                 x_array=X_train,
                                                 y_array=Y_train,
                                                 marker_dic=markers_dic,
                                                 used_in='training')

    # split X_temp and Y_temp into validation and test dataset
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp,
                                                    Y_temp,
                                                    test_size=0.5,
                                                    random_state=42,
                                                    stratify=Y_temp)

    # get original and synthetic markers for validation
    print(f'Total makers used for validation: {len(Y_val)}')
    markers_acc_subset1 = markers_acc[~markers_acc.Accession.isin(
        train_markers['accessions'])]

    val_markers = original_n_synthetic_markers(df=markers_acc_subset1,
                                               x_array=X_val,
                                               y_array=Y_val,
                                               marker_dic=markers_dic,
                                               used_in='validation')

    # get original and synthetic markers for validation
    print(f'Total makers used for test: {len(Y_test)}')

    markers_acc_subset2 = markers_acc_subset1[
        ~markers_acc_subset1.Accession.isin(val_markers['accessions'])]

    test_markers = original_n_synthetic_markers(df=markers_acc_subset2,
                                                x_array=X_test,
                                                y_array=Y_test,
                                                marker_dic=markers_dic,
                                                used_in='testing')
    # reconstituted markers df
    mdf = pd.concat([train_markers['segregated markers'],
                     val_markers['segregated markers'],
                     test_markers['segregated markers']], axis=0)
    if btype != 'unbalanced':
        mdf.to_csv(f'{exp}_original_and_synthetic_markers.tsv',
               sep='\t', index=False)
    mdf.drop(columns='compartment', inplace=True)
    return mdf



def my_train_test_split(df_data, train_size,
                        smote_type, neighbors, exp, utype, ucols):
    if utype == 'sml':
        tmt = df_data.filter(regex='^TMT').columns.to_list()
    else:
        tmt = ucols
    ndf_data_acc = df_data.loc[:, ['Accession'] + tmt + ['marker']]
    cols = tmt + ['marker']
    # spliting markers for training and test
    only_markers_acc = ndf_data_acc[ndf_data_acc.marker != 'unknown'].copy(
        deep=True)
    only_markers = only_markers_acc.loc[:, cols] # no accessions
    markers_acc = only_markers_acc.loc[:, ['Accession'] + cols]

    # encoding categorical labels into numeric ones
    dir_marker_dic, inv_marker_dic = markers_map(only_markers)
    only_markers['marker'] = only_markers['marker'].map(dir_marker_dic)
    # print(f'Only markers: {only_markers.columns.to_list()}')
    # # split into input and output elements

    x_0, y_0 = only_markers.iloc[:, : -1], only_markers.iloc[:, -1]

    # hot_encode the target variable
    y = LabelEncoder().fit_transform(y_0)

    starting_markers = len(y)
    # make sure there are enough markers to balance the sampling classes
    check_neighbors_and_sample(n_neighbors=neighbors, markers_list=y,
                               markers_dic=inv_marker_dic, exp=exp)

    # # transform the dataset to balance the sampling classes
    if smote_type == 'borderline':
        x, y = borderline_smote_estimation(x_0, y_0, neighbors)
    elif smote_type == 'over_under':
        x, y = over_under_smote(x_0, y_0, neighbors)
    elif smote_type == 'smote':
        oversample = SMOTE(k_neighbors=neighbors)
        x, y = oversample.fit_resample(x_0, y_0)

    else:
        x = x_0
        y = y_0

    if smote_type == 'unbalanced':
        # ----------------------------------------
        mdf = create_sets(x=x, y=y, train_size=train_size,
                          markers_acc=markers_acc, exp=exp,
                          markers_dic=inv_marker_dic,
                          btype='unbalanced')
        print(mdf.head())
        print(mdf.shape)
    else:
        final_markers = len(y)
        synthetic_markers = final_markers - starting_markers
        print(f'Total original markers: {starting_markers}')
        print(f'Total synthetic markers: {synthetic_markers}')
        print(f'Total original plus synthetic markers: {final_markers}')
        # # create training and test dataset for balanced classes
        # X_train, X_temp, Y_train, Y_temp = train_test_split(x,
        #                                                     y,
        #                                                     test_size=train_size,
        #                                                     random_state=42,
        #                                                     stratify=y)
        # print(f'Total makers used for training: {len(Y_train)}')
        # # get original and synthetic markers for training
        # train_markers = original_n_synthetic_markers(df=markers_acc,
        #                                              x_array=X_train,
        #                                              y_array=Y_train,
        #                                              marker_dic=inv_marker_dic,
        #                                              used_in='training')
        #
        # # split X_temp and Y_temp into validation and test dataset
        # X_val, X_test, Y_val, Y_test = train_test_split(X_temp,
        #                                                 Y_temp,
        #                                                 test_size=0.5,
        #                                                 random_state=42,
        #                                                 stratify=Y_temp)
        #
        # # get original and synthetic markers for validation
        # print(f'Total makers used for validation: {len(Y_val)}')
        # markers_acc_subset1 = markers_acc[~markers_acc.Accession.isin(
        #                                         train_markers['accessions'])]
        #
        # val_markers = original_n_synthetic_markers(df=markers_acc_subset1,
        #                                            x_array=X_val,
        #                                            y_array=Y_val,
        #                                            marker_dic=inv_marker_dic,
        #                                            used_in='validation')
        #
        # # get original and synthetic markers for validation
        # print(f'Total makers used for test: {len(Y_test)}')
        #
        # markers_acc_subset2 = markers_acc_subset1[
        #     ~markers_acc_subset1.Accession.isin(val_markers['accessions'])]
        #
        # test_markers = original_n_synthetic_markers(df=markers_acc_subset2,
        #                                             x_array=X_test,
        #                                             y_array=Y_test,
        #                                             marker_dic=inv_marker_dic,
        #                                             used_in='testing')
        # # reconstituted markers df
        # mdf = pd.concat([train_markers['segregated markers'],
        #                       val_markers['segregated markers'],
        #                       test_markers['segregated markers']], axis=0)
        # mdf.to_csv(f'{exp}_original_and_synthetic_markers.fitted.tsv',
        #            sep='\t', index=False)
        # mdf.drop(columns='compartment', inplace=True)
        mdf = create_sets(x=x, y=y, train_size=train_size,
                          markers_acc=markers_acc, exp=exp,
                          markers_dic=inv_marker_dic, btype='balanced')
    return inv_marker_dic,  dir_marker_dic, mdf



def data_for_prediction(df, utype):
    # full data without marker removal
    ndf = df.copy(deep=True)
    ndf = ndf[ndf['marker'] == 'unknown']
    del ndf['marker']
    if utype == 'sml':
        tmt = ndf.filter(regex='^TMT').columns.to_list()
        tmt_df = ndf.loc[:, tmt]
        tmt_label = ndf['Accession'].to_list()
        return {'tmt_data': tmt_df, 'tmt_accession': tmt_label}
    else: # deep learning holder
        x='holder'
        return #{'tmt_plus_data':tmt_df, 'tmt_plus_accession': tmt_label}



def prediction_on_data(model, unseen_data, accessions, pred_type,
                       thresholds, dirmarker):

    # Apply thresholds to the unseen data
    print('prediction on unseen data. Data size:', unseen_data.shape)

    prob_pred = predict_n_apply_thresholds(X_data=unseen_data,
                                           thresholds=thresholds,
                                           model=model)

    # reconstruct df dataset with predicted labels plus probs
    df_prob = reconstruct_df(pred=prob_pred['y_pred'],
                             prob=prob_pred['y_prob'],
                             marker_dic=dirmarker,
                             pred_type=pred_type,
                             accessions=accessions)
    return df_prob


def possible_duplicates(df, col, max_value, prediction_type, marker_dic):
    ndf = df.copy(deep=True)
    ndf = ndf[ndf[col] > max_value]
    if ndf.empty:
        return pd.DataFrame()
    else:
        ndf.index.name = 'Accession'
        ndf.reset_index(inplace=True)
        ndf.rename(columns=marker_dic, inplace=True)
        # m = f'Warning: {ndf.shape[0]} ambiguous predictions! '
        # m +='Classification will be done based on the highest '
        # m += f'probability value'
        # print(m)
        # ndf.to_csv(f'{prediction_type}.ambiguous_predictions.tsv', sep='\t',
        #            index=False)
        return ndf


def hyperparameter_tuning(x_train, y_train, mdl_type):
    if mdl_type == 'SVM':
        param = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001,
                           0.00001, 0.000001],
                 'kernel': ['rbf', 'linear'],
                 'class_weight': ['balanced', {0:1, 1:3}, {0:1, 1:5}],
                 'decision_function_shape': ['ovr']}

        grid = GridSearchCV(svm.SVC(probability=True),
                                    param_grid=param, cv=5,
                                    refit=True, scoring='f1_weighted',
                                    verbose=0)
    elif mdl_type == 'KNN':
        # create base classifier and calibrate probability
        base_knn = KNeighborsClassifier()
        calibrated_knn = CalibratedClassifierCV(estimator=base_knn)

        # create pipeline adding the calibratation as the final step
        pipeline = mp([('scaler', StandardScaler()),
                       ('classifier', calibrated_knn)])

        params = {'classifier__estimator__n_neighbors': np.arange(2, 30, 3),
                 'classifier__estimator__leaf_size': np.arange(2, 50, 5),
                 'classifier__estimator__weights': ['uniform', 'distance'],
                 'classifier__estimator__metric': ['minkowski'],
                 'classifier__estimator__p': [2]}
        calibration_parameters = {'classifier__method': ['sigmoid'],
                                  'classifier__cv': [5],
                                  'classifier__ensemble': [True]}
        param = {**params, **calibration_parameters}

        # cross-validation strategy
        kf = KFold(n_splits=4, shuffle=True, random_state=41)

        grid = GridSearchCV(estimator=pipeline,
                            param_grid=param,
                            cv=kf,
                            refit=True,
                            scoring='f1_weighted',
                            verbose=0,
                            n_jobs=-1)
    elif mdl_type == 'RF':
        # base classifier and calibrate probability
        base_rf = RandomForestClassifier()
        calibrated_rf = CalibratedClassifierCV(estimator=base_rf)

        # pipeline adding the calibration as the final step
        pipeline = mp([('scaler', StandardScaler()),
                       ('classifier', calibrated_rf)])

        params = {
            'classifier__estimator__n_estimators': randint(10, 500),
            # Number of trees
            'classifier__estimator__max_depth': [None, 10, 20, 30, 40],
            # Max depth of trees
            'classifier__estimator__min_samples_split': randint(2, 6),
            'classifier__estimator__min_samples_leaf': randint(1, 4),
            'classifier__estimator__max_features': ['sqrt', 'log2'],
            'classifier__estimator__class_weight': ['balanced'],
            'classifier__estimator__bootstrap': [True]}

        calibration_params = {'classifier__method': ['sigmoid'],
                              'classifier__cv': [5],
                              'classifier__ensemble': [True]}

        param = {**params, **calibration_params}

        # RandomizedSearchCV
        grid = RandomizedSearchCV(estimator=pipeline,
                                  param_distributions=param,
                                  n_iter=50,
                                  cv=5,
                                  scoring='f1_weighted',
                                  n_jobs=-1,
                                  verbose=1,
                                  random_state=41,
                                  return_train_score=True)
    else: # NB naive bayes
        params = {'classifier__estimator__var_smoothing': np.logspace(
                                                           0, -9, num=100),
                  'classifier__estimator__priors':[None]}
        calibration_params = {'classifier__method': ['sigmoid', 'isotonic'],
                              'classifier__cv': [5],
                              'classifier__ensemble': [True]}
        param = {**params, **calibration_params}

        base_nb = GaussianNB()
        calibrated_nb = CalibratedClassifierCV(estimator=base_nb)

        # pipeline adding the calibration as the final step
        pipeline = mp([('scaler', StandardScaler()),
                       ('classifier', calibrated_nb)])
        # define the grid search parameters
        cv_method = RepeatedStratifiedKFold(n_splits=3,
                                            n_repeats=3,
                                            random_state=41)
        grid = GridSearchCV(estimator=pipeline,
                            param_grid=param,
                            cv=cv_method,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=0)

    # fitting the model for grid search
    if mdl_type == 'SVM':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        grid.fit(x_train_scaled, y_train)
    else: # scaling is done in the 'pipeline' above
        grid.fit(x_train, y_train)

    # retun only parameters of interest with full name estructure
    best_params = parameters_of_interest(searched_parameters=param,
                                         estimated_parameters=grid)
    # Check if the pipeline is fitted
    print("Is pipeline fitted?:", hasattr(grid, 'best_estimator_'))
    return best_params


def parameters_of_interest(searched_parameters, estimated_parameters):
    full_params = estimated_parameters.best_estimator_.get_params()
    best_params = {}
    for k in searched_parameters.keys():
        if k in full_params.keys():
            best_params[k] = full_params[k]
        else:
            print('Missing hyperparameter', k)
    return best_params


def one_dim_2_hot_encoded(y_true, y_prob):
    n_classes = y_prob.shape[1]  # Get number of classes from probability matrix
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    y_true = y_true_onehot
    return y_true


def optimize_thresholds(y_true, y_prob, pred_type):
    '''
    Find optimal thresholds using Youden's J statistic and precision.
    '''
    n_classes = y_prob.shape[1]
    thresholds = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))
    f1 = np.zeros((n_classes,))
    dic = {}
    dic2 = {}
    # Convert 1D array to one-hot encoded format if needed
    if y_true.ndim == 1:
        y_true = one_dim_2_hot_encoded(y_true=y_true, y_prob=y_prob)

    for i in range(n_classes):
        # obtaining threshold with roc curve
        fpr, tpr, thresholds_roc = roc_curve(y_true[:, i], y_prob[:, i])
        j_statistic = tpr - fpr
        best_roc_threshold_idx = np.argmax(j_statistic)
        best_roc_threshold = thresholds_roc[best_roc_threshold_idx]
        # obtaining threshold with precision-recall pairs
        precision_i, recall_i, thresholds_prec = precision_recall_curve(y_true[:, i],
                                                                        y_prob[:, i])
        # adjust the threshold to give more weight to precision or recall
        # if recall is higher, then false negatives are minimized at the expense
        # of potentially lower precision:
        best_prec_threshold_idx = np.argmax(0.6*precision_i[:-1] +
                                            0.4*recall_i[:-1])
        best_prec_threshold = thresholds_prec[best_prec_threshold_idx]

        # use combined approach to find optimal threshold:
        thresholds[i] = (best_roc_threshold + best_prec_threshold) / 2
        y_pred_roc = (y_prob[:, i] >= thresholds[i]).astype(int)
        dic2[i] = list(y_pred_roc)
        precision[i] = precision_score(y_true[:, i],
                                       y_pred_roc,
                                       average='binary',
                                       zero_division=0)
        recall[i] = recall_score(y_true[:, i],
                                 y_pred_roc,
                                 average='binary',
                                 zero_division=0)
        f1[i] = f1_score(y_true[:, i],
                         y_pred_roc,
                         average='binary',
                         zero_division=0)
        dic[i]= {'precision': precision[i],
                 'recall': recall[i],
                 'f1-score': f1[i]}
    my_metrics = pd.DataFrame.from_dict(dic, orient='index')
    # relax thresholds by 10%
    orig_threshold_df = pd.DataFrame(thresholds, columns=['original threshold'])
    relaxed_thresholds = thresholds * 0.9
    relaxed_threshold_df = pd.DataFrame(relaxed_thresholds,
                                     columns=['10% relaxed threshold'])

    cat = pd.concat([orig_threshold_df, relaxed_threshold_df], axis=1).T
    print('Note thresholds have been relaxed by 10%:')
    # y_pred adjusted with thresholds
    y_pred_adj_df = pd.DataFrame(dic2)
    return {'y_pred_train.threshold': y_pred_adj_df ,
            f'{pred_type}.thresholds': cat,
            'relaxed thresholds': relaxed_thresholds,
            'metrics':my_metrics}


def multiclass_classification(X_train, y_train, X_val, y_val, model, pred_type):
    model.fit(X_train, y_train)
    # Get predicted labels for training data and their probabilities

    y_prob_train = model.predict_proba(X_train)
    optimal = optimize_thresholds(y_true=y_train,
                                  y_prob=y_prob_train,
                                  pred_type=pred_type)
    metrics1 = optimal['metrics']

    # evaluate validation set:
    prob_pred = predict_n_apply_thresholds(X_data=X_val,
                                           thresholds=optimal['relaxed thresholds'],
                                           model=model)

    metrics_2 = classification_report(y_true=y_val, y_pred=prob_pred['y_pred'],
                                      zero_division=0)
    return ({'model': model,
             'y_pred_train': optimal['y_pred_train.threshold'],
             'y_prob_train': y_prob_train,
             'y_prob_val': prob_pred['y_prob'],
             'y_pred_val': prob_pred['y_pred'],
             'thresholds': optimal['relaxed thresholds'],
             'thresholds_df': optimal[f'{pred_type}.thresholds']},
             (metrics1, metrics_2))


def predict_n_apply_thresholds(X_data, thresholds, model):
    # a runtime warning is triggered here when an entry has no prediction at all
    # this is handled by catching the warning but fillna with 0 later in the df
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                                category=RuntimeWarning,
                                message='invalid value encountered in divide')
        y_prob = model.predict_proba(X_data)
    y_pred = (y_prob >= thresholds).astype(int)
    return {'y_prob':y_prob, 'y_pred':y_pred}


def get_best_call(prob_array, pred_array, markers):
    df_prob = pd.DataFrame(prob_array)
    df_prob['max_value'] = df_prob.max(axis=1)
    # assig classification based on max probability value
    df_prob['raw classification'] = df_prob.idxmax(axis=1)
    df_prob['raw classification'] = df_prob['raw classification'].map(markers)
    df_pred = pd.DataFrame(pred_array)
    df_pred['total'] = df_pred.sum(axis=1)
    df_pred= df_pred.copy(deep=True)
    df_pred['new_column'] = df_pred['total'].apply(
        lambda x: 'unknown' if x == 0 else df_prob['raw classification'].iloc[
            df_pred[df_pred['total'] ==x].index[0]])
    merged = pd.merge(df_prob, df_pred, left_index=True, right_index=True,
                      how='outer')
    return merged



def revert_dummies(arrays_list, colnames_list, inv_dict):
    pairs = zip(arrays_list, colnames_list)
    series = []
    count = 0
    for array, colnames in pairs:
        count = count + 1
        df = pd.DataFrame(array, columns=colnames)
        df['total'] = df.sum(axis=1)
        colname = 'Pred label' if count == 1 else 'True label'
        df[colname] = np.where(df['total'] > 0, df.idxmax(axis=1), 'unknown')
        df[colname] = df[colname].map(inv_dict)
        df[colname] = df[colname].fillna('unknown')
        series.append(df[colname])
    merged_series = pd.concat(series, axis=1)
    return merged_series


def metrics_report(metrics1, metrics2, metrics3, markers, pred_type, dataset):
    all_metrics = {'train': format_report(metrics1, markers),
                   'validation': format_report(metrics2, markers),
                   'test': format_report(metrics3, markers)}
    _ = lopit_utils.write_to_excel(dfs_dic=all_metrics,
                                   outname=f'{pred_type}.'
                                           f'{dataset}.assessment.metrics')
    return 'Done'


def reconstruct_df(pred, prob, marker_dic, pred_type, accessions):
    if isinstance(pred, np.ndarray):
        df_pred = pd.DataFrame(pred, index=accessions)
    else: # pred is df
        df_pred = pred.copy(deep=True)
        df_pred.index = accessions
    if isinstance(prob, np.ndarray):
        df_prob = pd.DataFrame(prob, index=accessions)
    else: # pred is df
        df_prob = prob.copy(deep=True)
        df_prob.index = accessions

    # name index column
    df_pred.index.name = 'Accession'
    df_prob.index.name = 'Accession'

    # when an entry has no probability shown for any class: force fillna(0) See
    # predict_n_apply_thresholds function
    df_prob.fillna(0, inplace=True)
    # get maximum probability stored
    tmp_df = pd.DataFrame(accessions, columns=['Accession'])
    tmp_df.set_index('Accession', inplace=True)
    tmp_df[f'{pred_type}.probability'] = df_prob.max(axis=1)
    # get df_prob index of max probability
    tmp_df[f'{pred_type}.prediction.no.threshold'] = df_prob.idxmax(axis=1)

    # prediction df
    if df_pred.shape[1] > 1:
        df_pred[f'total {pred_type}'] = df_pred.sum(axis=1)
    else:
        df_pred.rename(columns={0: f'{pred_type}.prediction'}, inplace=True)

    # raise warning if ambiguous predictions are found
    if f'total {pred_type}' in df_pred.columns:
        adf = possible_duplicates(df=df_pred, col=f'total {pred_type}',
                                  max_value=1,
                                  prediction_type=pred_type,
                                  marker_dic=marker_dic)
        if not adf.empty:
            # catch proteins with probability to belong to more than one
            # compartment and select the compartment index with the highest
            # probability
            conditions = [(df_pred[f'total {pred_type}'] == 1),
                          (df_pred[f'total {pred_type}'] > 1)]
            choices = [df_pred.idxmax(axis=1),
                       df_prob.idxmax(axis=1)]
            df_pred[f'{pred_type}.prediction.threshold'] = np.select(conditions,
                                                           choices,
                                                           default='unknown')
            tmp_df[f'{pred_type}.prediction.threshold'] = df_pred[
                f'{pred_type}.prediction.threshold']
        else:
            df_pred[f'{pred_type}.prediction.threshold'] = np.where(
                                                df_pred[f'total {pred_type}'] == 1,
                                                df_pred.idxmax(axis=1),'unknown')
            tmp_df[f'{pred_type}.prediction.threshold'] = df_pred[
                f'{pred_type}.prediction.threshold']

    # map predictions
    c = [f'{pred_type}.prediction.no.threshold',
         f'{pred_type}.prediction.threshold']

    for col in c:
        tmp_df[col] = df_pred[f'{pred_type}.prediction.threshold'].map(
                                                lambda x: marker_dic.get(x, x))

    for ddf in [df_prob, df_pred]:
        # make Accession the first column
        ddf.reset_index(inplace=True)
        # name columns accordingly to the markers
        ddf.rename(columns=marker_dic, inplace=True)
    # join dfs
    df_prob_new = pd.merge(df_prob, tmp_df, left_on='Accession',
                           right_on='Accession', how='outer')
    return df_prob_new


def prepare_df(array, inv_dic, dir_dic):
    # create df from array
    df = pd.DataFrame(array)
    # map markers onto df
    df = df[0].map(inv_dic)
    # hot encode compatible with multiclass analysis
    df = pd.get_dummies(df).astype(int)
    # change col names for numeric indexes
    df.rename(columns=dir_dic, inplace=True)
    # sort columns from 0 - end to avoid label mixup during np array conversion
    df.sort_index(axis=1, inplace=True)
    # convert df to array for multiclass analysis compatibility
    array = df.values
    colnames = df.columns.tolist()
    return array, colnames


def deduplicate_keeping_indices(df, indices_to_keep):
    df = df.copy(deep=True)
    # get duplicates
    duplicates = df.duplicated(subset=df.columns.tolist(), keep=False)
    df['is_duplicate'] = duplicates
    dfd = df[duplicates]
    # get synthetic marker duplicate index names
    sm_duplicates = dfd[~dfd.index.isin(indices_to_keep)]
    # keep only non-duplicates sm
    df_sm = df[~df.index.isin(sm_duplicates.index.to_list() + indices_to_keep)]
    # from duplicates keep only those with index names present in
    # indices_to_keep Note: not all indexes are present in indices_to_keep
    dfd = dfd[dfd.index.isin(indices_to_keep)]
    # concatenate all non duplicate entries
    df_filtered_final = pd.concat([dfd, df_sm])
    del df_filtered_final ['is_duplicate']
    return df_filtered_final.reset_index()


def original_n_synthetic_markers(df, x_array, y_array, marker_dic, used_in):
    # add labels to sets
    y_df = pd.DataFrame(y_array, dtype=int, columns=['marker'])
    y_df['compartment'] = y_df['marker'].map(lambda x: marker_dic.get(x, x))
    # add y labels to x array
    x_df = pd.DataFrame(x_array)
    x_df['marker'] = y_df['marker'].to_list()
    x_df['compartment'] = y_df['compartment'].to_list()
    x_df['Accession'] = [f'SM-{x}' for x in range(x_df.shape[0])]

    # Note: df contains original accessions only, not SM accessions
    # create single markers df.
    # for compatibility: move marker to compartment and add encoded marker
    df = df.copy(deep=True)
    df.rename(columns={'marker': 'compartment'}, inplace=True)
    dir_dic = {v:k for k, v in marker_dic.items()}
    df['marker'] = df['compartment'].map(lambda x: dir_dic.get(x, x))
    #
    cat = pd.concat([df, x_df])
    cat = cat.copy(deep=True)
    cat.set_index('Accession', inplace=True)

    sep_markers = deduplicate_keeping_indices(cat, df['Accession'].to_list())
    sep_markers['used in'] = used_in
    accessions = [acc for acc in sep_markers['Accession'].to_list() if not
                  acc.startswith('SM-')]
    return {'segregated markers': sep_markers, 'accessions': accessions}


def extract_parameters(params_dic, sep):
    base_params = {
        k.replace(f'classifier{sep}estimator{sep}', ''): v
        for k, v in params_dic.items()
        if k.startswith(f'classifier{sep}estimator{sep}')}

    calibration_params = {
        k.replace(f'classifier{sep}', ''): v
        for k, v in params_dic.items()
        if k.startswith(f'classifier{sep}') and not k.startswith(
            f'classifier{sep}estimator{sep}')}
    return base_params, calibration_params


def calibrate_classifier(prediction_type, parameters, dataset):
    # extract parameters
    base_params, calibration_params = extract_parameters(params_dic=parameters,
                                                         sep='__')
    print(f'{dataset}-base parameters for {prediction_type}:\n', base_params)
    print(f'{dataset}-calibration parameters for {prediction_type}:\n',
          calibration_params)
    # create base classifier and apply parameters
    if prediction_type == 'KNN':
        base_classifier = KNeighborsClassifier(**base_params)
    elif prediction_type == 'RF':
        base_classifier = RandomForestClassifier(**base_params,
                                                 random_state=41)
    elif prediction_type == 'NB':
        base_classifier = GaussianNB(**base_params)
    else:
        base_classifier = None

    # calibrate classifier with calibration parameters
    calibrated_classifier = CalibratedClassifierCV(estimator=base_classifier,
                                                   **calibration_params)
    model = make_pipeline(StandardScaler(),
                          OneVsRestClassifier(calibrated_classifier))
    return model


def initialize_model(pred_type, params, dataset):
    if pred_type == 'SVM':
        model = make_pipeline(StandardScaler(),
                              OneVsRestClassifier(CalibratedClassifierCV(
            estimator=svm.SVC(probability=True, **params, random_state=41),
                              cv=5, method='sigmoid')))
    elif pred_type == 'KNN':
        model = calibrate_classifier(prediction_type=pred_type,
                                     parameters=params, dataset=dataset)
    elif pred_type == 'RF':
        model = calibrate_classifier(prediction_type=pred_type,
                                     parameters=params,
                                     dataset=dataset)
    else:
        model = calibrate_classifier(prediction_type=pred_type,
                                     parameters=params,
                                     dataset=dataset)
    return model


def prediction_workflow(model, dataset,
                        x_train, x_val, x_test,
                        y_train, y_val, y_test,
                        acc_train, acc_val, acc_test,
                        tmt_data, tmt_accession,
                        marker_dics, pred_type):

    inv_marker_dic, dic_marker = marker_dics
    # predictions
    if pred_type != 'NB':
        # hot encode as 2D for compatibility with ROC curve
        y_train, col_train = prepare_df(y_train, inv_marker_dic, dic_marker)
        y_val, col_val = prepare_df(y_val, inv_marker_dic, dic_marker)
        y_test, col_test = prepare_df(y_test, inv_marker_dic, dic_marker)

        predictions = train_val_test(model=model, prediction_type=pred_type,
                                     x_train=x_train, y_train=y_train,
                                     acc_train=acc_train,
                                     x_val=x_val, y_val=y_val,
                                     acc_val=acc_val,
                                     x_test=x_test, y_test=y_test,
                                     acc_test=acc_test,
                                     markers=inv_marker_dic,
                                     dataset=dataset)
    else:
        # 2D hot encode not done due to NB incompatibility with ROC curve
        predictions = train_val_test(model=model, prediction_type=pred_type,
                                        x_train=x_train, y_train=y_train,
                                        x_val=x_val, y_val=y_val,
                                        x_test=x_test, y_test=y_test,
                                        acc_train=acc_train,
                                        acc_val=acc_val,
                                        acc_test=acc_test,
                                        markers=inv_marker_dic,
                                        dataset=dataset)

    # df with train + validation + test sets, model and relaxed thresholds
    #cat, dic_train_val['model'], thresholds, thresholds_df
    training_sets, trained_model, thresholds, thresholds_df = predictions

    # predict classification for unseen data

    ndf = prediction_on_data(model=trained_model,
                             unseen_data=tmt_data,
                             accessions=tmt_accession,
                             pred_type=pred_type,
                             thresholds=thresholds,
                             dirmarker=inv_marker_dic)
    report_cols = ['Accession',
                   f'{pred_type}.probability',
                   f'{pred_type}.prediction.no.threshold',
                   f'{pred_type}.prediction.threshold']

    for_main_df = pd.concat([ndf.loc[:, report_cols],
                            training_sets.loc[:, report_cols]], axis=0)
    nndf = pd.concat([ndf, training_sets.loc[:, report_cols]], axis=0)
    return nndf, for_main_df, training_sets, thresholds_df


def remove_prefixes(dic, prefix_separator):
    new_dic = {}
    for key, value in dic.items():
        if '__' in key:
            new_key = key.split(prefix_separator)[1]
        else:
            new_key = key
        new_dic[new_key] = value
    return new_dic

def df_to_array(df, suffix, used_in):
    ndf = df.copy(deep=True)
    ndf = ndf[ndf['used in'] == used_in]
    accs = ndf.Accession.to_list()
    labels = np.array(ndf.marker.to_list())
    ndf.drop(columns=['Accession', 'marker', 'used in'], inplace=True)
    # colnames = ndf.columns.tolist() unmute only if needed
    return {f'x_{suffix}': ndf,
            f'y_{suffix}': labels,
            f'acc_{suffix}':accs}


def bespoke_classification(training_info, accuracy_threshold,
                       dataset, pred_type, tmt_data, tmt_accession, verbosity):
    print(f'*** {pred_type} information for {dataset} ***')
    # training information, inv and dir marker dics and marker df
    inv_marker_dic,  dir_marker_dic, markers_df = training_info
    train_info = df_to_array(df=markers_df, suffix='train', used_in='training')
    val_info = df_to_array(df=markers_df, suffix='val',  used_in='validation')
    test_info = df_to_array(df=markers_df, suffix='test', used_in='testing')

    # estimate best params
    best_params = hyperparameter_tuning(x_train=train_info['x_train'],
                                        y_train=train_info['y_train'],
                                        mdl_type=pred_type)
    if pred_type == 'SVM':
        print(f'{dataset}-best parameters for {pred_type}:\n {best_params}')

    #  define model: using the best parameters
    model = initialize_model(pred_type=pred_type, params=best_params,
                             dataset=dataset)

    # predictions
    pdata, pred_data, training_df, thresh = prediction_workflow(model=model,
                                                     dataset=dataset,
                                                     tmt_data=tmt_data,
                                                     tmt_accession=tmt_accession,
                                                     marker_dics=[inv_marker_dic,
                                                                 dir_marker_dic] ,
                                                     pred_type=pred_type,
                                                                **train_info,
                                                                **val_info,
                                                                **test_info)
    return {'prediction_data': pdata,
            'pred_data_df': pred_data,
            'training_df': training_df,
            'thresholds_df': thresh}


def convert_if_numeric(x):
    try:
        return int(x)
    except:
        return str(x)


def format_report(classif_report, inv_markers):
    if isinstance(classif_report, pd.DataFrame):
        classif_report = classif_report.reset_index(
            ).rename(columns={'index': 'marker'})
        classif_report['marker'] = classif_report[
            'marker'].map(lambda x: inv_markers.get(x, x))
        acc_df = classif_report
    else:
        rep = {'macro avg': 'macro_avg', 'weighted avg': 'weighted_avg',
               'micro avg': 'micro_avg', 'samples avg': 'samples_avg'}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        class_report = pattern.sub(lambda m: rep[re.escape(m.group(0))],
                                   classif_report)
        cr = class_report.split('\n')[2:-1]
        compartments = [line.split(' ') for line in cr
                        if not line.startswith('\n')]
        dic = {}
        for lista in compartments:
            values = [value.strip() for value in lista if value != '']
            if values:
                key = convert_if_numeric(values[0])
                if values[0] != 'accuracy':
                    dic[key] = [float(i) for i in values[1:]]
                else:
                    dic[key] = [' ', ' '] + [float(i) for i in values[1:]]
        acc_df = pd.DataFrame.from_dict(dic)
        acc_df = acc_df.T
        acc_df.rename(columns={0:'Precision', 1: 'Recall',
                               2: 'F1-score', 3: 'Support'}, inplace=True)
        acc_df.reset_index(inplace=True)
        acc_df.rename(columns={'index': 'marker'}, inplace=True)
        acc_df['marker'] = acc_df['marker'].map(lambda x:
                                                     inv_markers.get(x, x))
    return acc_df


def report(y_test, pred_labels, prediction_type, verbosity):
    # 'confusion matrix by classes
    cf_matrix = multilabel_confusion_matrix(y_test, pred_labels,
                                            labels=y_test)
    tmp = pd.DataFrame.from_dict(dict(zip(y_test, cf_matrix.tolist())),
                                 orient='index')
    tmp.rename(columns={0: 'True positives(TP)-False Negatives(FN)',
                        1: 'False positives(FP)-True Negatives(TN)'},
               inplace=True)
    cf_matrix_df = tmp[['True positives(TP)-False Negatives(FN)',
                        'False positives(FP)-True Negatives(TN)']].apply(
        lambda x: [v for lst in x for v in lst], axis=1, result_type="expand")
    colnames = {0: 'True positives(TP)', 1: 'False Negatives(FN)',
                2: 'False positives(FP)', 3: 'True Negatives(TN)'}
    cf_matrix_df.rename(columns=colnames, inplace=True)

    #  global and by class classifier accuracy
    #  note: if a category is not represented in the data apply 0 to
    #  precision (avoid average issue caused by zero division)
    class_report = classification_report(y_test, pred_labels, zero_division=0)
    report_df = format_report(class_report)
    accuracy = accuracy_score(y_test, pred_labels)
    # if verbosity:
    #     _ = write_xls(prediction_type, cf_matrix_df, report_df)
    return accuracy


def x_prediction_with_threshold(y_prob, thresholds):
    y_pred = (y_prob >= thresholds).astype(int)

    return y_pred


def multiclass_classification_nb(X_train, y_train, X_val,
                                 y_val, model, pred_type):
    model.fit(X_train, y_train)
    # Get predicted labels for training data and their probabilities
    # y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)

    optimal = optimize_thresholds(y_true=y_train,
                                  y_prob=y_prob_train,
                                  pred_type=pred_type)
    metrics1 = optimal['metrics']

    # evaluate validation set:
    prob_pred = predict_n_apply_thresholds(X_data=X_val,
                                           thresholds=optimal['relaxed thresholds'],
                                           model=model)

    y_pred_val = prob_pred['y_pred']
    y_pred_val_df = pd.DataFrame(y_pred_val)
    prob_pred.update({'y_pred': y_pred_val_df})
    y_pred_val_ = np.argmax(prob_pred['y_pred'], axis=1)

    metrics_2 = classification_report(y_val, y_pred_val_, zero_division=0)

    return ({'model': model,
             'y_pred_train': optimal['y_pred_train.threshold'],
             'y_prob_train': y_prob_train,
             'y_prob_val': prob_pred['y_prob'],
             'y_pred_val': prob_pred['y_pred'],
             'thresholds': optimal['relaxed thresholds'],
             'thresholds_df': optimal[f'{pred_type}.thresholds']},
             (metrics1, metrics_2))


def train_val_test(model, prediction_type,
                   x_train, y_train, acc_train,
                   x_val, y_val, acc_val,
                   x_test, y_test, acc_test,
                   markers, dataset):
    if prediction_type == 'NB':
        # Train on train set and validation
        train_val = multiclass_classification_nb(X_train=x_train,
                                                 y_train=y_train,
                                                 X_val=x_val,
                                                 y_val=y_val,
                                                 model=model,
                                                 pred_type=prediction_type)
    else:
        # Train on train set and validation
        train_val = multiclass_classification(X_train=x_train,
                                              y_train=y_train,
                                              X_val=x_val,
                                              y_val=y_val,
                                              model=model,
                                              pred_type=prediction_type)

    dic_train_val, metrics_train_val = train_val
    thresholds_df = dic_train_val['thresholds_df']
    thresholds_df.rename(columns=markers, inplace=True)
    metrics_1, metrics_2 = metrics_train_val
    thresholds = np.array(dic_train_val['thresholds'])

    # reconstruct training dataset with true and pred labels plus probs
    # Note: dic_train_val['y_pred_train'] == prediction under optimal thresholds
    rec_df_train = reconstruct_df(pred=dic_train_val['y_pred_train'],
                                  prob=dic_train_val['y_prob_train'],
                                  marker_dic=markers,
                                  pred_type=prediction_type,
                                  accessions=acc_train)

    # reconstruct validation dataset with true and predicted labels plus probs
    rec_df_val = reconstruct_df(pred=dic_train_val['y_pred_val'],
                                prob=dic_train_val['y_prob_val'],
                                marker_dic=markers,
                                pred_type=prediction_type,
                                accessions=acc_val)
    # apply thresholds to the test set
    prob_pred_test = predict_n_apply_thresholds(X_data=x_test,
                                           thresholds=thresholds,
                                           model=dic_train_val['model'])
    if prediction_type == 'NB':
        y_test_pred = np.argmax(prob_pred_test['y_pred'], axis=1)
    else:
        y_test_pred = prob_pred_test['y_pred']

    # calculate metrics
    metrics_3 = classification_report(y_true=y_test,
                                      y_pred=y_test_pred,
                                      zero_division=0)
    # reconstruct test dataset with true and predicted labels plus probs
    rec_df_test = reconstruct_df(pred=prob_pred_test['y_pred'],
                                 prob=prob_pred_test['y_prob'],
                                 marker_dic=markers,
                                 pred_type=prediction_type,
                                 accessions=acc_test)
    # write metrics to a file
    _ = metrics_report(metrics1=metrics_1, metrics2=metrics_2,
                       metrics3=metrics_3, markers=markers,
                       pred_type=prediction_type, dataset=dataset)

    cat = pd.concat([rec_df_train, rec_df_val, rec_df_test])
    return cat, dic_train_val['model'], thresholds, thresholds_df


def changing_preexisting_colnames(df, markers_info):
    pre_existing = ['marker',
                    'SVM.probability',
                    'SVM.prediction.no.threshold',
                    'SVM.prediction.threshold',
                    'KNN.probability',
                    'KNN.prediction.no.threshold',
                    'KNN.prediction.threshold',
                    'RF.probability',
                    'RF.prediction.no.threshold',
                    'RF.prediction.threshold',
                    'NB.probability',
                    'NB.no.threshold',
                    'NB.prediction.threshold',
                    'most.common.pred.supported.by',
                    'best.pred.supported3+marker',
                    'best.pred.supported4+marker',
                    'most.common.pred.SVM.KNN.RF.NB.hdbscan',
                    'most.common.pred.SVM.KNN.RF.NB']
    joined = '\n'.join(pre_existing)
    cols = df.columns.to_list()
    if 'best.pred.supported3+marker' in cols:
        print(f'*** The following columns already exist in the input '
              f'dataframe and will be renamed:\n{joined}\n'
              f'\nIf you have multiple supervised machine learning '
              f'classifications\nin the master file, you MUST remove '
              f'them before making any\nnew classification '
              f'as column rename is only supported once ***')
        new = {k: f'{k}_base' for k in pre_existing}
        df.rename(columns=new, inplace=True)
        merged_df = pd.merge(df, markers_info, on='Accession', how='left')
        return merged_df
    else:
        if 'marker' in cols and 'best.pred.supported3+marker' not in cols:
            m = list(set(df['marker'].to_list()))
            if len(m) == 1:
                del df['marker']
                merged_df = pd.merge(df, markers_info, on='Accession',
                                     how='left')
                return merged_df
            else:
                return df
        elif 'marker' not in cols:
            merged_df = pd.merge(df, markers_info, on='Accession', how='left')
            return merged_df
        else:
            return None


def supervised_clustering(directory, odf, markers_df, train_size,
                          accuracy_threshold, smote_type, neighbors,
                          outname, accessory_file, verbosity):
    # combination
    comb = os.path.split(directory)[-1]
    # create and move into the new directory
    try:
        os.mkdir(directory)
    except:
        print('Directory already exists')
        pass

    dataset = os.path.split(directory)[-1]
    print(dataset)
    os.chdir(directory)
    # sml classification
    print('Beginning of supervised learning classification ...')
    if isinstance(markers_df, pd.DataFrame):
        df = marker_detection(odf, markers_df)
    else:
        df = marker_detection(odf, markers_df[comb])

    # create training, validation, and test sets
    print('Generating training, validation, and test sets ...')
    tvt_sets = my_train_test_split(df_data=df,
                                   train_size=train_size,
                                   smote_type=smote_type,
                                   neighbors=neighbors,
                                   exp=dataset,
                                   utype='sml',
                                   ucols=[])

    # unseen data for supervised machine learning
    tmt_info = data_for_prediction(df=df, utype='sml')

    # Support vector machine classification
    svm = bespoke_classification(training_info=tvt_sets,
                                 accuracy_threshold=accuracy_threshold,
                                 dataset=dataset,
                                 pred_type='SVM',
                                 tmt_data=tmt_info['tmt_data'],
                                 tmt_accession=tmt_info['tmt_accession'],
                                 verbosity=verbosity)
    svm_df, svm_training_df = svm['pred_data_df'], svm['training_df']
    full_svm_df = svm['prediction_data']


    # K-nearest neighbors classification
    knn = bespoke_classification(training_info=tvt_sets,
                                 accuracy_threshold=accuracy_threshold,
                                 dataset=dataset,
                                 pred_type='KNN',
                                 tmt_data=tmt_info['tmt_data'],
                                 tmt_accession=tmt_info['tmt_accession'],
                                 verbosity=verbosity)

    knn_df, knn_training_df = knn['pred_data_df'], knn['training_df']
    full_knn_df= knn['prediction_data']

    # Random forest classification
    rf = bespoke_classification(training_info=tvt_sets,
                                accuracy_threshold=accuracy_threshold,
                                dataset=dataset,
                                pred_type='RF',
                                tmt_data=tmt_info['tmt_data'],
                                tmt_accession=tmt_info['tmt_accession'],
                                verbosity=verbosity)
    rf_df, knn_training_df = rf['pred_data_df'], rf['training_df']
    full_rf_df = rf['prediction_data']
    
    # Naive Bayes classification
    nb = bespoke_classification(training_info=tvt_sets,
                                accuracy_threshold=accuracy_threshold,
                                dataset=dataset,
                                pred_type='NB',
                                tmt_data=tmt_info['tmt_data'],
                                tmt_accession=tmt_info['tmt_accession'],
                                verbosity=verbosity)
    nb_df, nb_training_df = nb['pred_data_df'], nb['training_df']
    full_nb_df = nb['prediction_data']

    #  all thresholds to a file
    all_thresholds = {'SVM': svm['thresholds_df'].T,
                      'KNN': knn['thresholds_df'].T,
                      'RF': rf['thresholds_df'].T,
                      'NB': nb['thresholds_df'].T}
    _ = lopit_utils.write_to_excel(dfs_dic=all_thresholds,
                                   outname=f'{dataset}.prediction_thresholds')

    # all predictions to a file
    all_predictions = {'SVM': full_svm_df,
                       'KNN': full_knn_df,
                       'RF': full_rf_df,
                       'NB': full_nb_df}
    oname = f'{dataset}.SVM.KNN.RF.NB.probabilities'
    _ = lopit_utils.write_to_excel(dfs_dic=all_predictions, outname=oname)

    #  construct a df with predictions
    reconsted_df = reduce(lambda left, right: pd.merge(left, right,
                                                     on='Accession',
                                                     how='outer'),
                        [svm_df, knn_df, rf_df, nb_df])
    if accessory_file is not None:
        acc_file = lopit_utils.accesion_checkup(odf, accessory_file,
                                                ftype='accessory file')
    else:
        acc_file = ''

    if isinstance(markers_df, pd.DataFrame):
        _ = wrapping_up(df, reconsted_df, markers_df, outname, acc_file)
    else:
        _ = wrapping_up(df, reconsted_df, markers_df[comb], outname, acc_file)

    return 'Done'''


def marker_checkup(entry, balancing_method, mtype, avail_dirs):
    if entry is not None and os.path.isfile(entry):
        markers_map = pd.read_csv(entry, sep='\t', header=0)
        cnames = markers_map.columns.to_list()
        if mtype == 'global':
            if 'Accession' not in cnames or 'marker' not in cnames:
                print("Marker file does not contain a column identified as "
                      "'Accession' or 'marker', or both. Exiting program ...")
                sys.exit(-1)
        else:
            if 'Accession' not in cnames:
                print("Marker file does not contain a column identified "
                      "as 'Accession'. Exiting program ...")
                sys.exit(-1)
            else:
                cols = [col for col in cnames if col != 'Accession']
                avail_cols = [os.path.split(d)[-1] for d in avail_dirs]
                intersect = set(cols).intersection(set(avail_cols))
                if (len(avail_cols) == len(cols) and
                    len(intersect) == len(avail_cols)):
                    pass
                else:
                    mc = ', '.join(cols)
                    ec = ', '.join(avail_cols)
                    print(f'Expected marker columns are:\n{ec}.\n'
                          f'Declared marker columns are:\n{mc}. '
                          f'Exiting program ...')
                    sys.exit(-1)
    else:
        print('Declared marker file does NOT exist. Exiting program...')
        sys.exit(-1)

    if mtype == 'global':
        mmap = markers_format(markers_map, balancing_method, dataset='global')
        return mmap
    else:
        dfs = {col: markers_map.loc[:, ['Accession', col]]
               for col in markers_map.columns.to_list() if col != 'Accession'}
        new_dic = {}
        for combination in dfs.keys():
            ren = dfs[combination].rename(columns={combination: 'marker'})
            new_dic[combination] = ren

        dfs_checked = {}
        for combination in new_dic.keys():
            mmap = markers_format(new_dic[combination], balancing_method,
                                  dataset=combination)
            dfs_checked[combination] = mmap
        return dfs_checked


def markers_format(markersmap, balancing_method, dataset):
    markersmap.marker.replace(' |-', '_', regex=True, inplace=True)
    vals = list(markersmap['marker'].unique())
    vals.remove('unknown')
    vals = natsorted(vals)
    if len(vals) > 50:
        print(f'marker type {dataset}\n:Markers infile contains over 50 unique '
              f'entries (markers + unknown).\nColor pallete for over 50 entries'
              'is not currently supported in lopit_utils.\n'
              'Exiting program...')
        sys.exit(-1)
    marker_labels = sorted(Counter(markersmap['marker']).items(),
                           key=lambda x: x[1], reverse=False)
    # major warning for the marker sizes
    d = {tup[0]: tup[1] for tup in marker_labels}
    l = {tup: d[tup] for tup in d.keys() if d[tup] < 6 if tup != 'unknown'}
    if l:
        m = ('***   WARNING 1  ***\n'
             f'Marker type {dataset}:\nThere are not enough markers for a '
             f'good prediction:\n{l}\nAt least 6 markers by compartment should '
             f'be declared. This program will continue if there are at least 3 '
             'markers per class (compartment) but predictions may be highly '
             'inaccurate when data is largely unbalanced.\n'
             '*** WARNING 2 ***: \n'
             'This program will fail due to lack of markers per class '
             'during:\n1) unbalanced method: if there are not enough markers '
             'for stratification, then additional markers must be provided for '
             'this method or a balance method should be selected, or\n'
             '2) balanced method: during smote in the Experiment subset if '
             'samples to fit <= than the number n_neighbors:\n'
             'case: n_neighbors = 3, n_samples_fit = 2, n_samples = 2\n'
             'then it is best to eliminate the offender markers and re-run '
             'the program.\n ********** end of WARNING **********')
        print(m)

        if [l[i] for i in l.keys() if l[i] < 3]:  # if the list is not empty
            m = f'{dataset}:\n. Exit program due to lack of markers:\n{l}'
            print(m)
    else:
        msize = {tup: d[tup] for tup in d.keys() if d[tup] > 50
                 if tup != 'unknown'}
        if balancing_method != 'unbalanced' and len(msize) > 0:
            m = ('***   WARNING   ***\n'
                 f'{dataset}:\nThere are more than 50 many markers at least '
                 f'in one class/comparment\n{msize}\n'
                 f'Note: \n'
                 '1) The unbalanced method might be very inaccurate\n'
                 '2) Specify the unbalanced method for classification\n'
                 'Exiting program...\n'
                 '*** end of WARNING   ***')
            print(m)
            sys.exit(-1)
        else:
            print(f'{dataset}:\nMarkers were checked and program will continue')
    return markersmap


def same_markers(df1, df2):
    df1 = df1.loc[(df1['marker'] == 'unknown')]
    df2 = df2.loc[(df2['marker'] == 'unknown')]
    master_dic = dict(zip(df1['Accession'], df1['marker']))
    marker_dic = dict(zip(df2['Accession'], df2['marker']))
    return master_dic == marker_dic


def format_markers(df):
    df['marker'] = df['marker'].str.replace(r' ', '_', regex=False)
    return df


def marker_detection(masterdf, markerdf):
    if not markerdf.empty:
        markerdf = format_markers(markerdf)
        if 'marker' in masterdf.columns.to_list():
            marker_df = lopit_utils.accesion_checkup(masterdf, markerdf)
            if same_markers(masterdf, marker_df):
                del masterdf['marker']
            else:
                masterdf.rename(columns={'marker': 'marker_old'},
                                inplace=True)
            new_master = pd.merge(masterdf, marker_df,
                                  on='Accession', how='left')
            new_master['marker'] = new_master['marker'].fillna('unknown')
            return new_master
        else:
            print('no marker column in marker file')
            sys.exit(-1)
    else:
        if 'marker' in masterdf.columns.to_list():
            return masterdf
        else:
            print('A marker file must be provided as no markers '
                  'are present in the TMT file.\nExiting program...')
            sys.exit(-1)


def wrapping_up(master_df, fdf, marker_df, outname, accessory_file):
    if 'hdb_labels_euclidean_TMT_pred_marker' in master_df.columns.to_list():
        hdbscan = True
    else:
        hdbscan = False

    if not marker_df.empty:  # changing colnames for previous sml predictions
        if 'marker_old' in master_df.columns.to_list():
            # fdf cols: 'Accession', 'SVM.prediction', 'KNN.prediction',
            # 'Random.forest.prediction', 'Naive.Bayes'
            intersect_cols = set(fdf.columns.to_list()).intersection(
                set(master_df.columns.to_list()))
            if intersect_cols:
                master_df.rename(columns={col: f'{col}_old' for col in
                                          fdf.columns.to_list()
                                          if col != 'Accession'},
                                 inplace=True)
        # # writing sml predictions (old and new ones)
        # ffdf = write_mydf([master_df, fdf],
        #                   outname, hdbscan, '', accessory_file)
    # writing sml predictions:
    _ = lopit_utils.write_mydf([master_df, fdf], outname,
                                  hdbscan, '', accessory_file)

    # return to main directory
    os.chdir('../..')
    return 'Done'


def traverse(infile, fileout, f_identificator, markers_file,
             markers_type, balance_method, verbosity):

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
    available_dirs = [f.path for f in os.scandir(infile) if f.is_dir()]
    print('available dirs', available_dirs)

    # checking markers
    if markers_type == 'global':
        markers_df = marker_checkup(markers_file,
                                    balance_method,
                                    mtype='global',
                                    avail_dirs='')
    else:
        markers_df = marker_checkup(markers_file,
                                    balance_method,
                                    mtype='by_combination',
                                    avail_dirs=available_dirs)
    #  create new dirs and respective dataframes
    target_files = {}
    missing_target_files = []
    for f in available_dirs:
        dir_name = os.path.split(f)[-1]
        print('---  working on  ---', dir_name)
        fpath = os.path.join(os.path.abspath(infile), f)
        fpath = os.path.join(fpath, f_identificator)
        tfiles = glob.glob(f'{fpath}*.tsv')
        print('current argument', tfiles)

        if len(tfiles) == 1:
            tf = tfiles[0]
        else:
            m = (f'There are no files or there are multiple target files '
                 f'(ambiguous): directory: {dir_name}, target {tfiles}.\n'
                 f'The recognition_motif must be unambiguous.\n')
            print(m)
            sys.exit(-1)
        # os.chdir(f)
        if os.path.isfile(tf):
            infile = os.path.abspath(tf)
            dfin = pd.read_csv(infile, sep='\t', header=0)
            #  check that loaded file corresponds to the correct combination
            fbname = os.path.split(infile)[-1]
            fbname = fbname.split('.')[0]
            bname = fbname.split(f_identificator)[-1]
            if bname != dir_name:
                print(f'Dataset combination inferred from directory name '
                      f'{dir_name} and input file {bname} do not match.'
                      f'file identificator declared is: {f_identificator} '
                      'Exiting program...')
                sys.exit(-1)

            npath = os.path.join(newdir, dir_name)
            if isinstance(markers_df, pd.DataFrame):
                # merge df and markers
                dfin_markers_df = changing_preexisting_colnames(dfin,
                                                                markers_df)
            elif isinstance(markers_df, dict):
                # check header is an existing combination in dir names
                # merge combination_df with markers
                try:
                    dfin_markers_df = changing_preexisting_colnames(dfin,
                                                                    markers_df[
                                                                     bname])
                except KeyError:
                    header = ', '.join(markers_df.keys())
                    print(f'Declared header in input file for {bname}'
                          f'is incorrect. Columns available in input file are: '
                          f'{header}. Exiting program...')
                    sys.exit(-1)
            else:
                print('Something is wrong with the markers file:',
                      markers_df)
                sys.exit(-1)
            # validate marker number by input dataset or combination:
            print('Checking marker numbers by combination')
            sliced_df = dfin_markers_df.loc[:, ['Accession', 'marker']]
            _ = markers_format(sliced_df, balance_method, dataset=dir_name)
            # integration of data in main dic after validation
            target_files[npath] = dfin_markers_df

        else:
            missing_target_files.append(f)
        # os.chdir('..')
    # stop program if at least one file is missing for the declared input dir
    if missing_target_files:
        m = f'File(s) is/are missing from: {missing_target_files}'
        m += 'exiting program'
        print(m)
        sys.exit(-1)
    return target_files, markers_df


def parallel_prediction(dic_with_dfs, balance_method,
                        markers_df, afile, verbosity):
    if afile is not None and os.path.isfile(afile):
        acc_file = pd.read_csv(afile, sep='\t', header=0)
    else:
        acc_file = None

    #  clustering each dataset in parallel
    try:
        njobs = os.cpu_count() - 1
        with Parallel(n_jobs=njobs, return_as='generator') as parallel:
            tasks = (delayed(supervised_clustering)
                                       (directory, dic_with_dfs[directory],
                                        markers_df,
                                        train_size=0.35,
                                        accuracy_threshold=0.90,
                                        smote_type=balance_method,
                                        neighbors=2, outname='SML',
                                        accessory_file=acc_file,
                                        verbosity=verbosity)
                                for directory in dic_with_dfs.keys())
            # execute tasks and force completion
            list(parallel(tasks))

    except ValueError as ve:
        print(ve)
        print('Exiting program...')
        sys.exit(-1)
    except Exception as e:
        print(e)
        print('Exiting program...')
        sys.exit(-1)
    finally:
        # Force garbage collection to clean up resources
        import gc
        gc.collect()
    return


#  ---   execute   ---   #


if __name__ == '__main__':
    #f = "df_PLNPLOPL2PL1.test.tsv"
    f = "Final_df_PLNPLOPL2PL1_SML.AccessoryInfo.tsv"
    master_df = pd.read_csv(f, sep='\t', header=0)
    marker_file = "Pmar_369markers.19354.20062024.tsv"
    mf = pd.read_csv(marker_file, sep='\t', header=0)

    # fdf, marker_df = supervised_clustering('directory', master_df,
    #                                        marker_file,
    #                                        train_size=0.33,
    #                                        accuracy_threshold=0.90,
    #                                        smote_type='borderline',
    #                                        neighbors=2, outname='new')
    # # writing sml predictions with previous sml prediction
    # _ = wrapping_up(master_df, fdf, marker_df, 'newdataset')
    # dfs_dic = traverse(infile, fileout, f_identificator)
    # results = parallel_prediction(dfs_dic, 'borderline',
    #                               marker_file, 'new')