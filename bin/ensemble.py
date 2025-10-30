import sys
import warnings
import pandas as pd
from numpy import unique
import overfitting as of
from sklearn.svm import SVC
import calibration_selection as cal_sel
from lopit_utils import parameters_of_interest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


warnings.filterwarnings('ignore')


meta_params = {
    'RF': {'final_estimator__n_estimators': [50, 75, 100, 125, 150,
                                              175, 200],
           'final_estimator__max_depth': [None, 10, 20, 30, 40, 50],
           'final_estimator__min_samples_split': [2, 3, 5],
           'final_estimator__min_samples_leaf': [1, 2, 3, 4, 6, 8],
           'final_estimator__max_features': ['log2', 'sqrt', 0.3, 0.5],
           'final_estimator__class_weight': ['balanced'],
           'final_estimator__bootstrap': [True]}}


def get_best_params(params, X_train, y_train, n_jobs, splits=5):
    base_model = {'SVM':SVC(),
                  'RF':RandomForestClassifier()}
    best_params_by_model = {}

    for classifier, params in params.items():
        model = base_model[classifier]
        cv_strategy = StratifiedKFold(n_splits=splits,
                                      shuffle=True,
                                      random_state=41)
        grid = GridSearchCV(model,
                            param_grid=params,
                            cv=cv_strategy,
                            scoring=['neg_log_loss', 'accuracy'],
                            refit='neg_log_loss',
                            n_jobs=n_jobs,
                            pre_dispatch='2*n_jobs',
                            verbose=1,
                            return_train_score=True)

        grid.fit(X_train, y_train)
        best_params_by_model[classifier] = grid.best_params_

    return best_params_by_model


def create_ensemble(X_train, y_train, X_val, y_val, metaclassifier,
                    n_jobs, calibration, params=None):

    # initialize classifiersm using best parameters from gridsearch
    if params is None: # grid search done in main
        params = get_best_params(params, X_train, y_train, n_jobs, splits=5)
    print('params are:\n', params)
    svm_clf = SVC(**params['SVM'],
                  random_state=42)
    #rf_clf = RandomForestClassifier(**params['RF'],
                                    # random_state=42)

    print(f'{'='*60}\nCreating ensemble by stacking SMV and RF.\n'
          f'Ensemble is using CV...\n{'='*60}')
    print(f'Meta estimator is: {metaclassifier}')

    if metaclassifier == 'RF':
        metaclassifier = RandomForestClassifier()
        metaparams = meta_params['RF']
    else:
        metaclassifier = None
        metaparams = None

    sclf = StackingClassifier(estimators=[('SVM', svm_clf)],
                              final_estimator=metaclassifier,
                              cv=5,
                              n_jobs=n_jobs)

    grid = GridSearchCV(estimator=sclf,
                        param_grid=metaparams,
                        cv=5,
                        scoring=['neg_log_loss', 'accuracy'],
                        refit='neg_log_loss',
                        n_jobs=n_jobs,
                        pre_dispatch='2*n_jobs',
                        verbose=0,
                        return_train_score=True)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # check for possible overfitting
    _ = of.possible_overfitting(grid=grid,
                                model_name='StackClassifier')

    # best parameters
    best_params = parameters_of_interest(searched_parameters=params,
                                         estimated_parameters=grid)

    best_params_df = pd.DataFrame.from_dict(best_params,
                                            orient='index',
                                            columns=['Value'])

    best_params_df.index.name = 'Parameter'
    best_params_df['Classifier'] = 'StackMetaclassifier-> RF'

    # Get cross-validation training accuracy
    cv_results = pd.DataFrame(grid.cv_results_)
    best_index = grid.best_index_

    train_accuracy = cv_results.loc[best_index, 'mean_train_accuracy']


    if calibration:
        unique_classes, class_counts = unique(y_val, return_counts=True)
        min_class_size = min(class_counts)

        if min_class_size <= 3:
            print(f'Calibration set has classes with <= 3 samples and CV can not proceed. '
                  f'Minimum class size: {min_class_size}.\n'
                  f'Exiting program.')
            sys.exit(1)

        # calibrate the classifier
        calibrated_model, x_val_proba  = cal_sel.calibrated_classifier(
                                                        best_model=best_model,
                                                        X_val=X_val,
                                                        y_val=y_val,
                                                        handle_imbalance=False)

        # validation accuracy for comparison
        val_accuracy = cv_results.loc[best_index, 'mean_test_accuracy']
        # Create stats dictionary
        stats = {'accuracy_train': train_accuracy,
                 'accuracy_val': val_accuracy,
                 'train_std': cv_results.loc[best_index, 'std_train_accuracy'],
                 'val_std': cv_results.loc[best_index, 'std_test_accuracy']}
        print(f'CV Training Accuracy: {train_accuracy:.4f} ± {stats['train_std']:.4f}')
        print(f'CV Validation Accuracy: {val_accuracy:.4f} ± {stats['val_std']:.4f}')
        return calibrated_model, best_params_df, x_val_proba, stats
    else:
        x_val_proba = None
        # Create stats dictionary
        stats = {'accuracy_train': train_accuracy,
                 'train_std': cv_results.loc[best_index, 'std_train_accuracy']}
        print(f'CV Training Accuracy: {train_accuracy:.4f} ± {stats['train_std']:.4f}')
        return best_model, best_params_df, x_val_proba, stats



