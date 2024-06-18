import os
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from functools import reduce
from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from natsort import index_natsorted, natsorted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, KFold,
                                     RepeatedStratifiedKFold)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)


sml_cols = ['SVM.prediction', 'KNN.prediction', 'Random.forest.prediction',
            'Naive.Bayes', 'most.common.pred.SVM.KNN.RF.NB.hdbscan',
            'most.common.pred.SVM.KNN.RF.NB', 'most.common.pred.supported.by',
            'best.pred.supported3+', 'best.pred.supported4+']


def predef_message(ml_method, acc, req_acc):
    m = (f'******** WARNING  {ml_method} ********\n'
         f'Accuracy is {acc} which is BELOW the requested accuracy '
         f'threshold of {req_acc}\n'
         '*************************\n')
    print(m)
    return 'Done'


def my_train_test_split(df_data, train_size):
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
    accs_short = set(all_markers).difference(set(accs))

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
    return x_train, x_test, y_train, y_test


def data_for_prediction(df):
    ndf = df.copy(deep=True)
    tmt = ndf.filter(regex='^TMT').columns.to_list()
    df_no_markers = ndf[ndf.marker == 'unknown']
    tmt_df = df_no_markers.loc[:, tmt]
    tmt_label = df_no_markers['Accession'].to_list()
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


def svm_classification(df, train_size, accuracy_threshold):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df,
                                                           train_size)
    tmt_data, tmt_label = data_for_prediction(df)
    # Define model
    # tuning C and gamma
    best_params = hyperparameter_tuning(x_train, y_train, 'SVM')
    # apply best parameters to model
    model = make_pipeline(StandardScaler(),
                          svm.SVC(kernel='rbf',
                                  C=best_params['C'],
                                  gamma=best_params['gamma'],
                                  probability=True,
                                  random_state=1))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = report(y_test, pred, 'SVM')
    # predict classification for whole dataset without entries with markers
    ndf = prediction_on_data(model, tmt_data, tmt_label,
                             'SVM.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('SVM', accuracy, accuracy_threshold)
    return ndf


def report(y_test, pred, prediction_type):
    # cf_matrix = confusion_matrix(y_test, pred)
    #  if a category is not represented in the data apply 0 to
    #  precision (avoid average issue caused by zero division)
    class_report = classification_report(y_test, pred, zero_division=0)
    accuracy = accuracy_score(y_test, pred)
    with open(f'{prediction_type}_classification_report.txt', 'w') as O:
        O.write(class_report)
    return accuracy


def knn_classification(df, train_size, accuracy_threshold):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df, train_size)
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
    accuracy = report(y_test, pred, 'KNN')
    # predict classification for whole dataset excluding entries with markers
    ndf = prediction_on_data(knn_model, tmt_data, tmt_label,
                             'KNN.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('KNN', accuracy, accuracy_threshold)
    return ndf


def random_forest_classification(df, train_size, accuracy_threshold):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df, train_size)
    tmt_data, tmt_label = data_for_prediction(df)
    # best params: the algorith kept failing due to string to float in markers.
    # Hence, it is hyperparemeter tuning is not implemented for rf
    #  define the model
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier())
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    accuracy = report(y_test, pred, 'Random_forest')
    # predict classification for whole dataset without entries with markers
    ndf = prediction_on_data(rf, tmt_data, tmt_label,
                             'Random.forest.prediction')
    #  ---
    if accuracy < accuracy_threshold:
        _ = predef_message('Random Forest',
                           accuracy, accuracy_threshold)
    return ndf


def naive_bayes_classifier(df, train_size, accuracy_threshold):
    # create training and test sets
    x_train, x_test, y_train, y_test = my_train_test_split(df, train_size)
    tmt_data, tmt_label = data_for_prediction(df)
    # best params
    best_params = hyperparameter_tuning(x_train, y_train, 'Naive Bayes')
    # Define model
    gnb = GaussianNB(var_smoothing=best_params['var_smoothing'])
    gnb.fit(x_train, y_train)
    pred = gnb.predict(x_test)
    accuracy = report(y_test, pred, 'Naive_Bayes')
    # predict classification for whole dataset without entries with markers
    ndf = prediction_on_data(gnb, tmt_data, tmt_label,
                             'Naive.Bayes')
    # ---
    if accuracy < accuracy_threshold:
        _ = predef_message('Naive Bayes',
                           accuracy, accuracy_threshold)
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


def common_prediction(df, cols, hdbscan=False, cutoff=''):
    ndf = df.copy(deep=True)
    if hdbscan is True:
        if 'hdb_labels_euclidean_TMT_pred_marker' in ndf.columns.to_list():
            cols = cols + ['hdb_labels_euclidean_TMT_pred_marker']

        if 'hdb_labels_euclidean_TMT_pred_marker' in cols:
            ml, size, sign = 'hdbscan', 5, '+hdbscan'
        else:
            ml, size, sign = '', 4, '+'
    else:
        ml, size, sign = 'tagmap', 5, '+tagmap'
        cols = cols + [f'tagm.map.allocation.cutoff_{cutoff}']

    ndf.set_index('Accession', inplace=True)
    wdf = ndf.loc[:, cols]
    dic1, dic2, dic3, dic4 = {}, {}, {}, {}
    for acc in wdf.index:
        y = wdf.loc[acc, :].values.flatten().tolist()
        temp = sorted(Counter(y).items(), key=lambda x: x[1], reverse=True)
        if len(temp) == 1:
            dic1[acc] = temp[0][0]
            dic2[acc] = temp[0][1]
            if temp[0][1] >= 3:
                if temp[0][1] >= 4:
                    dic4[acc] = temp[0][0]
                    dic3[acc] = temp[0][0]
                else:
                    if acc not in dic3.keys():
                        dic3[acc] = temp[0][0]
        elif 1 < len(temp) < size:
            if temp[0][1] > temp[1][1]:
                dic1[acc] = temp[0][0]
                dic2[acc] = temp[0][1]
                if temp[0][1] >= 3:
                    if temp[0][1] >= 4:
                        dic4[acc] = temp[0][0]
                        dic3[acc] = temp[0][0]
                    else:
                        if acc not in dic3.keys():
                            dic3[acc] = temp[0][0]
            if size == 5:
                if temp[0][1] == temp[1][1] == temp[2][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0], temp[2][0]])
                    dic2[acc] = 0
                elif temp[0][1] == temp[1][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0]])
                    dic2[acc] = 0
            elif size == 4:
                if temp[0][1] == temp[1][1]:
                    dic1[acc] = '|'.join([temp[0][0], temp[1][0]])
                    dic2[acc] = 0
        else:
            dic1[acc] = 'unknown'
            dic2[acc] = 0
    if size == 5:
        df['most.common.pred.SVM.KNN.RF.NB.hdbscan'] = df['Accession'].map(dic1)
    else:
        df['most.common.pred.SVM.KNN.RF.NB'] = df['Accession'].map(dic1)

    complete = [f'most.common.pred.SVM.KNN.RF.NB.{ml}',
                f'most.common.pred.supported.by{sign}',
                f'best.pred.supported3{sign}marker',
                f'best.pred.supported4{sign}marker']
    dics = [dic1, dic2, dic3, dic4]

    for col, dic in zip(complete, dics):
        df[col] = df['Accession'].map(dic)
        df[col] = df[col].fillna('unknown')
    df[complete] = df.loc[:, complete].fillna('unknown')
    return df


def supervised_clustering(odf, markers_file, train_size, accuracy_threshold):
    if markers_file != '':
        markers_df = marker_checkup(markers_file)
        df = changing_preexisting_colnames(odf, markers_df)
    else:
        markers_df = pd.DataFrame()
        df = odf
    # sml classification
    svm_df = svm_classification(df, train_size, accuracy_threshold)
    knn_df = knn_classification(df, train_size, accuracy_threshold)
    randomf_df = random_forest_classification(df, train_size,
                                              accuracy_threshold)
    nb_df = naive_bayes_classifier(df, train_size, accuracy_threshold)
    reconst_df = reduce(lambda left, right: pd.merge(left, right,
                                                     on='Accession',
                                                     how='outer'),
                        [svm_df, knn_df, randomf_df, nb_df])
    return reconst_df, markers_df


def write_df(dfs_list, outname, hdbscan, cutoff):
    df = reduce(lambda left, right: pd.merge(left, right,
                                             on='Accession',
                                             how='left'), dfs_list)
    fill_cols = ['SVM.prediction', 'KNN.prediction',
                 'Random.forest.prediction', 'Naive.Bayes']
    if hdbscan is True and cutoff == '':
        markers = df[df.marker != 'unknown']
        markers_dic = dict(zip(markers.Accession, markers.marker))
        for col in fill_cols:
            df[col] = df[col].fillna(df['Accession'].map(markers_dic))

    # compute shared predictions and prepare final df
    final_df = common_prediction(df, fill_cols, hdbscan, cutoff)

    # write final df
    dataset = final_df['Dataset'].to_list()[0]
    fpath = os.path.join(os.getcwd(), f'Final_df_{dataset}.{outname}.'
                                      f'Supervised.ML.tsv')
    if outname != 'tagm_added':
        final_df.to_csv(fpath, sep='\t', index=False)
    return final_df


def marker_checkup(entry):
    if entry is not None and os.path.isfile(entry):
        markers_map = pd.read_csv(entry, sep='\t', header=0)
        cnames = markers_map.columns.to_list()
        if 'Accession' not in cnames or 'marker' not in cnames:
            print('marker file does not contain a column identified '
                  'as Accession or marker, or both')
            sys.exit(-1)
        markers_map.marker.replace(' |-', '_', regex=True, inplace=True)
        vals = list(markers_map['marker'].unique())
        vals.remove('unknown')
        vals = natsorted(vals)
        if len(vals) > 50:
            print('Markers infile contains over 50 unique entries '
                  '(markers + unknown).\nColor pallete for over 50 entries is '
                  'not currently supported in lopit_utils (line 28).\n'
                  'Exiting program...')
            sys.exit(-1)
        return markers_map
    else:
        print('Declared marker file does NOT exist. Exiting program...')
        sys.exit(-1)


#  ---   execute   ---   #


if __name__ == '__main__':
    #f = "df_PLNPLOPL2PL1.test.tsv"
    f = "Final_df_PLNPLOPL2PL1.dataset.Supervised.ML.tsv"
    master_df = pd.read_csv(f, sep='\t', header=0)
    marker_file = pd.read_csv('markers_test.tsv',
                              sep='\t', header=0)

    fdf, marker_df = supervised_clustering(master_df, marker_file,
                                           train_size=0.7,
                                           accuracy_threshold=0.90)
    # writing sml predictions with previous sml prediction
    if not marker_df.empty:
        ffdf = write_df([master_df, fdf, marker_df],
                        'newdataset', True, '')
    # writing di novo sml prediction without any previous sml prediction
    else:
        ffdf = write_df([master_df, fdf], 'newdataset',
                        True, '')

