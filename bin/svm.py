import sys
import pandas as pd
from numpy import unique
from sklearn import svm
import overfitting as of
from sklearn.pipeline import Pipeline as mp
import calibration_selection as cal_sel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lopit_utils import parameters_of_interest



def svm_hyperparams(X_train, y_train, X_val, y_val, n_jobs, calibration):
    print(f'{'='*60}\nHyperparameter tuning for SVM ...\n{'='*60}')

    base_svm = svm.SVC()
    # pipeline
    pipeline = mp([('classifier', base_svm)])


    cv_strategy = StratifiedKFold(n_splits=5,
                                  shuffle=True,
                                  random_state=42)

    param = {'classifier__C': [0.0625, 0.125, 0.25, 0.5,
                               1, 2, 4, 8, 16, 20],
             'classifier__gamma': ['scale', 'auto', 0.001,
                                   0.01, 0.1, 1, 10, 100],
             'classifier__kernel': ['rbf', 'linear'],
             'classifier__class_weight': ['balanced'],
             'classifier__decision_function_shape': ['ovr'],
             'classifier__probability': [True]}#,
             # 'classifier__max_iter': [100]


    grid = GridSearchCV(pipeline,
                        param_grid=param,
                        cv=cv_strategy,
                        scoring=['neg_log_loss', 'accuracy'],
                        refit='neg_log_loss',
                        n_jobs=n_jobs,
                        pre_dispatch='2*n_jobs',
                        verbose=1,
                        return_train_score=True)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # check for possible overfitting
    _ = of.possible_overfitting(grid=grid, model_name='SVM')

    # best parameters
    best_params = parameters_of_interest(searched_parameters=param,
                                         estimated_parameters=grid)

    best_params_df = pd.DataFrame.from_dict(best_params,
                                            orient='index',
                                            columns=['Value'])
    best_params_df.index.name = 'Parameter'
    best_params_df['Classifier'] = 'SVM'

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
                                                        handle_imbalance=True)
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


def get_svm_decision_scores(model, X):
    # Extract the SVM classifier from pipeline

    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        print('in pipeline')
        svm_classifier = model.named_steps['classifier']
    else:
        svm_classifier = model

    # Get decision function scores
    decision_scores = svm_classifier.decision_function(X)
    return decision_scores