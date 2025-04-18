import glob
import os
import re
import sys
import lopit_utils
import numpy as np
import pandas as pd
from sklearn import svm
from functools import reduce
from collections import Counter
from joblib import Parallel, delayed
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from natsort import natsorted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.model_selection import (GridSearchCV, KFold,
                                     RepeatedStratifiedKFold, train_test_split)
from sklearn.preprocessing import (StandardScaler, PowerTransformer,
                                   LabelEncoder)
from sklearn.metrics import (classification_report, multilabel_confusion_matrix,
                             accuracy_score)


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



def unbalanced_train_test_split(df_data, train_size):
    ndf_data = df_data.copy(deep=True)
    # -----------------------------------------
    # spliting markers for training and test
    only_markers = ndf_data[ndf_data.marker != 'unknown']
    accs = pd.DataFrame(only_markers.groupby('marker')[
                         'Accession'].sample(frac=train_size,
                                             random_state=1))
    # marker accessions
    all_markers = only_markers.Accession.to_list()
    accs_large = accs.Accession.to_list()
    accs_short = set(all_markers).difference(set(accs_large))

    # marker subsets
    test_markers = ndf_data[ndf_data.Accession.isin(accs_short)]
    train_markers = ndf_data[ndf_data.Accession.isin(accs_large)]
    # ----------------------------------------
    # spliting main data into train and test
    # data and labels
    tmt = ndf_data.filter(regex='^TMT').columns.to_list()
    x_train = train_markers.loc[:, tmt]
    y_train = train_markers.loc[:, 'marker']
    x_test = test_markers.loc[:, tmt]
    y_test = test_markers.loc[:, 'marker']
    # _ = print_labels(only_markers, y_train, y_test)
    return x_train, x_test, y_train, y_test


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


def my_train_test_split(df_data, train_size,
                        smote_type, neighbors, exp):
    tmt = df_data.filter(regex='^TMT').columns.to_list()
    ndf_data = df_data.loc[:, tmt + ['marker']]

    # spliting markers for training and test
    only_markers = ndf_data[ndf_data.marker != 'unknown'].copy(deep=True)
    # encoding categorical labels into numeric ones
    dir_marker_dic, inv_marker_dic = markers_map(only_markers)
    only_markers['marker'] = only_markers['marker'].map(dir_marker_dic)
    # # split into input and output elements
    x, y = only_markers.iloc[:, : -1], only_markers.iloc[:, -1]
    # label encode the target variable
    y = LabelEncoder().fit_transform(y)

    # make sure there are enough markers to balance the sampling classes
    check_neighbors_and_sample(n_neighbors=neighbors, markers_list=y,
                               markers_dic=inv_marker_dic, exp=exp)

    # # transform the dataset to balance the sampling classes
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-clas
    # sification/
    if smote_type == 'borderline':
        x, y = borderline_smote_estimation(x, y, neighbors)
    elif smote_type == 'over_under':
        x, y = over_under_smote(x, y, neighbors)
    elif smote_type == 'smote':
        oversample = SMOTE(k_neighbors=neighbors)
        x, y = oversample.fit_resample(x, y)
    else:
        pass

    if smote_type == 'unbalanced':
        print('***WARNING unbalanced data will be split: 70% markers will '
              'be used for traning and the remaining 30% for evaluation')
        x_train, x_test, y_train, y_test = unbalanced_train_test_split(df_data,
                                                                       0.7)
    else:
        # create training and test dataset for balanced classes
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=train_size, random_state=42)
        # return numeric labels to categorical labels
        y_train = [inv_marker_dic.get(i, i) for i in y_train]
        y_test = [inv_marker_dic.get(i, i) for i in y_test]
    return x_train, x_test, y_train, y_test


def data_for_prediction(df):
    # full data without marker removal
    ndf = df.copy(deep=True)
    tmt = ndf.filter(regex='^TMT').columns.to_list()
    tmt_df = df.loc[:, tmt]
    tmt_label = df['Accession'].to_list()
    return tmt_df, tmt_label


def prediction_on_data(model, data, data_label, prediction_type):
    if prediction_type == 'Naive Bayes':
        data_transformed = PowerTransformer().fit_transform(data)
        whole_pred = model.predict(data_transformed)
    else:
        whole_pred = model.predict(data)
    pdf = pd.DataFrame(zip(whole_pred, data_label),
                       columns=[f'{prediction_type}', 'Accession'])
    return pdf


def hyperparameter_tuning(x_train, y_train, mdl_type):
    if mdl_type == 'SVM':
        param = {'C': [0.1, 1, 10, 100, 1000],
                 'gamma': [1, 0.1, 0.01, 0.001, 0.0001,
                           0.00001, 0.000001],
                 'kernel': ['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid=param, refit=True, verbose=0)
    elif mdl_type == 'KNN':
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        param = {'n_neighbors': np.arange(2, 30, 1),
                 'leaf_size': np.arange(2, 50, 1)}
        grid = GridSearchCV(KNeighborsClassifier(),
                            param_grid=param, cv=kf, refit=True, verbose=0)
    else:
        param = {'var_smoothing': np.logspace(0, -9, num=100)}
        # define the grid search parameters
        cv_method = RepeatedStratifiedKFold(n_splits=4,
                                            n_repeats=3,
                                            random_state=999)
        grid = GridSearchCV(GaussianNB(), param_grid=param, cv=cv_method,
                            scoring='accuracy', verbose=0)
        data_transformed = PowerTransformer().fit_transform(x_train)

    # fitting the model for grid search
    if mdl_type == 'SVM' or mdl_type == 'KNN':
        grid.fit(x_train, y_train)
    else:
        grid.fit(data_transformed, y_train)
    return grid.best_params_


def calculate_probabilities(model, pred_labels, x_train, x_test,
                            y_train, y_test, marker_dic):
    # predict calibrated probabilities
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated.fit(x_train, y_train)
    label_probs = calibrated.predict_proba(x_test)[:, 1]
    inverted_dic = {marker_dic[i]: i for i in marker_dic.keys()}
    tmp_df = x_test.copy(deep=True)  # test data
    tmp_df['True'] = y_test  # true labels
    tmp_df['Predicted'] = pred_labels # predicted labels
    tmp_df['True'] = tmp_df['True'].map(inverted_dic)
    tmp_df['Predicted'] = tmp_df['Predicted'].map(inverted_dic)
    tmp_df['probability'] = label_probs
    # tmp_df.to_csv('temporary_svm_classification.tsv', sep='\t', index=False)
    return tmp_df


def svm_classification(df, train_size, accuracy_threshold,
                       smote_type, neighbors, dataset, verbosity):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df,
                                                           train_size,
                                                           smote_type,
                                                           neighbors,
                                                           exp=dataset)
    tmt_data, tmt_label = data_for_prediction(df)
    # Define model
    # tuning C and gamma
    best_params = hyperparameter_tuning(x_train, y_train, 'SVM')
    # apply best parameters to model
    model = make_pipeline(StandardScaler(),
                          svm.SVC(kernel='rbf',
                                  C=best_params['C'],
                                  gamma=best_params['gamma'],
                                  probability=False,
                                  random_state=1))
    model.fit(x_train, y_train)
    pred_labels = model.predict(x_test)
    # svc_roc_auc = roc_auc_score(y_test, pred_labels, multi_class='ovr')
    # print('ROC', svc_roc_auc)
    # _ = calculate_probabilities(model, pred_labels, x_train,
    #                             x_test, y_train, y_test,
    #                             marker_dic) # see comment below:
    # the calculated probabilities when labels match are too small. This is
    # problem might be related to the small dataset size. see predict_proba
    # SVC documentation

    accuracy = report(y_test, pred_labels, 'SVM', verbosity)
    # predict classification for whole dataset
    ndf = prediction_on_data(model, tmt_data, tmt_label,
                             'SVM.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('SVM', accuracy, accuracy_threshold,
                           dataset)
    else:
        print(f'***   SVM estimated accuracy for {dataset} is {accuracy}   ***')
    return ndf


def format_report(classificatior_report):
    rep = {'macro avg': 'macro_avg', 'weighted avg': 'weighted_avg'}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    class_report = pattern.sub(lambda m: rep[re.escape(m.group(0))],
                               classificatior_report)
    cr = class_report.split('\n')[2:-1]
    compartments = [line.split(' ') for line in cr
                    if not line.startswith('\n')]

    dic = {}
    for lista in compartments:
        values = [value for value in lista if value != '']
        if values:
            if values[0] != 'accuracy':
                dic[values[0]] = values[1:]
            else:
                dic[values[0]] = [' ', ' '] + values[1:]

    acc_df = pd.DataFrame.from_dict(dic, orient='index',
                                    columns=['precision', 'recall',
                                             'f1-score', 'support'])
    acc_df = acc_df.reset_index().rename(columns={'index': 'marker'})
    return acc_df


def write_xls(prediction_type, df1, df2):
    with (pd.ExcelWriter(f'{prediction_type}.accuracy.estimations.xlsx')
          as out):
        df1.to_excel(out, sheet_name='confusion_matrix')
        df2.to_excel(out, sheet_name='classification_report')
        return 'Done'


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
    if verbosity:
        _ = write_xls(prediction_type, cf_matrix_df, report_df)
    return accuracy


def knn_classification(df, train_size, accuracy_threshold,
                       smote_type, neighbors, dataset, verbosity):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df,
                                                           train_size,
                                                           smote_type,
                                                           neighbors)
    tmt_data, tmt_label = data_for_prediction(df)
    # best params
    best_params = hyperparameter_tuning(x_train, y_train, 'KNN')
    #  define the model
    knn_model = make_pipeline(StandardScaler(),
                              KNeighborsClassifier(algorithm='auto',
                                                   leaf_size=best_params[
                                                       'leaf_size'],
                                                   metric='minkowski',
                                                   p=2,  # 2 euclidian distance
                                                   metric_params=None,
                                                   n_jobs=1,
                                                   n_neighbors=best_params[
                                                       'n_neighbors'],
                                                   weights='uniform'))
    knn_model.fit(x_train, y_train)
    pred = knn_model.predict(x_test)
    accuracy = report(y_test, pred, 'KNN', verbosity)
    # predict classification for whole dataset
    ndf = prediction_on_data(knn_model, tmt_data, tmt_label,
                             'KNN.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('KNN', accuracy, accuracy_threshold,
                           dataset)
    else:
        print(f'***   KNN estimated accuracy for {dataset} is {accuracy}   ***')
    return ndf


def random_forest_classification(df, train_size, accuracy_threshold,
                                 smote_type, neighbors, dataset, verbosity):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df,
                                                           train_size,
                                                           smote_type,
                                                           neighbors)
    tmt_data, tmt_label = data_for_prediction(df)
    # best params: the algorith kept failing due to string to float in markers.
    # Hence, hyperparemeter tuning is not implemented for rf
    #  define the model
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier())
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    accuracy = report(y_test, pred, 'Random_forest', verbosity)
    # predict classification for whole dataset
    ndf = prediction_on_data(rf, tmt_data, tmt_label,
                             'Random.forest.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('Random Forest',
                           accuracy, accuracy_threshold, dataset)
    else:
        print(f'***   Random Forest estimated accuracy for {dataset} is '
              f'{accuracy}   ***')
    return ndf


def naive_bayes_classifier(df, train_size, accuracy_threshold,
                           smote_type, neighbors, dataset, verbosity):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df,
                                                           train_size,
                                                           smote_type,
                                                           neighbors)
    tmt_data, tmt_label = data_for_prediction(df)
    # best params
    best_params = hyperparameter_tuning(x_train, y_train, 'Naive Bayes')
    # Define model
    gnb = GaussianNB(var_smoothing=best_params['var_smoothing'])
    gnb.fit(x_train, y_train)
    pred = gnb.predict(x_test)
    accuracy = report(y_test, pred, 'Naive_Bayes', verbosity)
    # predict classification for whole dataset
    ndf = prediction_on_data(gnb, tmt_data, tmt_label,
                             'Naive.Bayes')
    # ---
    if accuracy < accuracy_threshold:
        _ = predef_message('Naive Bayes',
                           accuracy, accuracy_threshold, dataset)
    else:
        print(f'***   Naive Bayes estimated accuracy for {dataset} '
              f'is {accuracy}   ***')
    return ndf


def changing_preexisting_colnames(df, markers_info):
    pre_existing = ['marker', 'SVM.prediction', 'KNN.prediction',
                    'Random.forest.prediction', 'Naive.Bayes',
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
              f'\nIf you have multiplesupervised machine learning '
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
        if 'marker' not in cols:
            merged_df = pd.merge(df, markers_info, on='Accession',
                                 how='left')
            return merged_df


def supervised_clustering(directory, odf, markers_df, train_size,
                          accuracy_threshold, smote_type, neighbors,
                          outname, accessory_file, verbosity):
    # combination
    comb = os.path.split(directory)[-1]
    # create and move into new directory
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

    svm_df = svm_classification(df, train_size, accuracy_threshold, smote_type,
                                neighbors, dataset, verbosity)
    knn_df = knn_classification(df, train_size, accuracy_threshold, smote_type,
                                neighbors, dataset, verbosity)
    randomf_df = random_forest_classification(df, train_size,
                                              accuracy_threshold, smote_type,
                                              neighbors, dataset, verbosity)
    nb_df = naive_bayes_classifier(df, train_size, accuracy_threshold,
                                   smote_type, neighbors, dataset, verbosity)
    reconst_df = reduce(lambda left, right: pd.merge(left, right,
                                                     on='Accession',
                                                     how='outer'),
                        [svm_df, knn_df, randomf_df, nb_df])
    #
    if accessory_file is not None:
        acc_file = lopit_utils.accesion_checkup(odf, accessory_file,
                                                ftype='accessory file')
    else:
        acc_file = ''

    if isinstance(markers_df, pd.DataFrame):
        _ = wrapping_up(df, reconst_df, markers_df, outname, acc_file)
    else:
        _ = wrapping_up(df, reconst_df, markers_df[comb], outname, acc_file)

    # _ = wrapping_up(df, reconst_df, markers_df, outname, acc_file)
    return 'Done'


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
        m = ('***   WARNING   ***\n'
             f'Marker type {dataset}:\nThere are not enough markers for a '
             f'good prediction:\n{l}\nAt least 6 markers by compartment should '
             f'be declared. This program will continue if there are at least 3 '
             'markers per compartment but predictions may be highly '
             'inaccurate when data is largely unbalanced. Warning: the '
             'program will fail during smote in the Experiment subset if '
             'samples to fit <= than the number n_neighbors:\n'
             'case: n_neighbors = 3, n_samples_fit = 2, n_samples = 2\n'
             'then it is best to eliminate the offender markers and re-run '
             'the program *******************')
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
    ffdf = lopit_utils.write_mydf([master_df, fdf], outname,
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

    #  clustering each dataset in parallel   #
    _ = Parallel(n_jobs=-1)(delayed(supervised_clustering)
                                   (directory, dic_with_dfs[directory],
                                    markers_df,
                                    train_size=0.33,
                                    accuracy_threshold=0.90,
                                    smote_type=balance_method,
                                    neighbors=2, outname='SML',
                                    accessory_file=acc_file,
                                    verbosity=verbosity)
                            for directory in dic_with_dfs.keys())
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