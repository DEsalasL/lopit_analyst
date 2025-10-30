import sys
import pandas as pd
from numpy import unique
import overfitting as of
import calibration_selection as cal_sel
from sklearn.pipeline import Pipeline as mp
from lopit_utils import parameters_of_interest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def rf_hyperparams(X_train, y_train, X_val, y_val, n_jobs,
                   calibration):
    print(f'{'='*60}\nHyperparameter tuning for RF ...\n{'='*60}')
    # base classifier and calibrate probability
    base_rf = RandomForestClassifier()

    # pipeline
    pipeline = mp([('classifier', base_rf)])

    params = {
        'classifier__n_estimators': [50, 75, 100, 125, 150,
                                     175, 200],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': [2, 3, 5],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 6, 8],
        'classifier__max_features': ['log2', 'sqrt', 0.3, 0.5],
        'classifier__class_weight': ['balanced'],
        'classifier__bootstrap': [True]}

    cv_hyperparams = StratifiedKFold(n_splits=5,
                                shuffle=True,
                                random_state=41)
    # GridSearchCV
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        cv=cv_hyperparams,
                        scoring=['neg_log_loss', 'accuracy'],
                        refit='neg_log_loss',
                        n_jobs=n_jobs,
                        pre_dispatch='2*n_jobs',
                        verbose=1,
                        return_train_score=True)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # check for possible overfitting
    _ = of.possible_overfitting(grid=grid, model_name='RF')

    # best parameters
    best_params = parameters_of_interest(searched_parameters=params,
                                         estimated_parameters=grid)
    best_params_df = pd.DataFrame.from_dict(best_params,
                                            orient='index',
                                            columns=['Value'])
    best_params_df.index.name = 'Parameter'
    best_params_df['Classifier'] = 'RF'

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





