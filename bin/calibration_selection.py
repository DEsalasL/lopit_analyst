import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight



def calibrated_classifier(best_model, X_val, y_val, handle_imbalance):

    print('Calibrating the Classifier ...')

    # Compute sample weights for imbalanced data
    if handle_imbalance:
        # Get unique classes and compute class weights
        classes = np.unique(y_val)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_val)
        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights = compute_sample_weight(class_weight_dict, y_val)
        print(f"Class distribution: {np.bincount(y_val)}")
        print(f"Computed class weights: {class_weight_dict}")
    else:
        sample_weights = None


    # Cross-validation for comparing methods to avoid leakage
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)

    isotonic_scores = []
    sigmoid_scores = []
    # initialize arrays to collect probabilities
    val_probs_isotonic = np.zeros((X_val.shape[0], len(np.unique(y_val))))
    val_probs_sigmoid = np.zeros((X_val.shape[0], len(np.unique(y_val))))

    # Cross-validation to compare calibration methods
    for train_idx, test_idx in cv_strategy.split(X_val, y_val):
        X_cal_train, X_cal_test = X_val[train_idx], X_val[test_idx]
        y_cal_train, y_cal_test = y_val[train_idx], y_val[test_idx]

        if handle_imbalance:
            # Weights for y_cal_train in current fold
            train_classes = np.unique(y_cal_train)
            train_class_weights = compute_class_weight('balanced',
                                                       classes=train_classes,
                                                       y=y_cal_train)
            train_class_weight_dict = dict(zip(train_classes, train_class_weights))
            fold_sample_weights = compute_sample_weight(train_class_weight_dict, y_cal_train)

            # Weights for y_cal_test in current fold
            test_classes = np.unique(y_cal_test)
            test_class_weights = compute_class_weight('balanced',
                                                      classes=test_classes,
                                                      y=y_cal_test)
            test_class_weight_dict = dict(zip(test_classes, test_class_weights))
            test_weights = compute_sample_weight(test_class_weight_dict, y_cal_test)
        else:
            fold_sample_weights = None
            test_weights = None

        # Isotonic calibration
        calibrated_isotonic = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model),
            method='isotonic',
            cv=2)
        calibrated_isotonic.fit(X_cal_train, y_cal_train,
                                sample_weight=fold_sample_weights)
        prob_isotonic = calibrated_isotonic.predict_proba(X_cal_test)
        val_probs_isotonic[test_idx] = prob_isotonic

        # Sigmoid calibration
        calibrated_sigmoid = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model),
            method='sigmoid',
            cv=2)
        calibrated_sigmoid.fit(X_cal_train, y_cal_train,
                               sample_weight=fold_sample_weights)
        prob_sigmoid = calibrated_sigmoid.predict_proba(X_cal_test)
        val_probs_sigmoid[test_idx] = prob_sigmoid

        # Scores for current fold
        brier_isotonic= brier_score(y_cal_test, prob_isotonic,
                                    sample_weight=test_weights)
        brier_sigmoid = brier_score(y_cal_test, prob_sigmoid,
                                    sample_weight=test_weights)

        log_loss_isotonic = log_loss(y_cal_test, prob_isotonic,
                                     sample_weight=test_weights)
        log_loss_sigmoid = log_loss(y_cal_test, prob_sigmoid,
                                    sample_weight=test_weights)

        # Store combined scores: lower score is better
        isotonic_scores.append((brier_isotonic + log_loss_isotonic) / 2)
        sigmoid_scores.append((brier_sigmoid + log_loss_sigmoid) / 2)

    # Calculate mean scores
    mean_isotonic = np.mean(isotonic_scores)
    mean_sigmoid = np.mean(sigmoid_scores)

    # Select best method
    best_method = 'isotonic' if mean_isotonic < mean_sigmoid else 'sigmoid'
    print(f"Best calibration method: {best_method}")

    # apply best calibration method
    cv_strategy = StratifiedKFold(n_splits=3,
                                  shuffle=True,
                                  random_state=42)

    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(best_model),
                                            method=best_method,
                                            cv=cv_strategy)

    # # Fit with sample weights was removed due to UserWarning below:
    # if handle_imbalance:
    #     calibrated_model.fit(X_val, y_val,
    #                          sample_weight=sample_weights)
    # else:
    #     calibrated_model.fit(X_val, y_val)
    # UserWarning: Since FrozenEstimator does not appear to accept sample_weight,
    # sample weights will only be used for the calibration itself.
    # Be warned that the result of the calibration is likely to be incorrect.
    calibrated_model.fit(X_val, y_val)
    gc.collect()
    print('Calibration complete.')
    # returning unbiased probabilities for validation set
    if best_method == 'isotonic':
        return calibrated_model, val_probs_isotonic
    else:
        return calibrated_model, val_probs_sigmoid


def brier_score(y_cal_test, prob_method, sample_weight=None):
    if prob_method.shape[1] == 2:
        score = brier_score_loss(y_cal_test, prob_method[:, 1],
                                 sample_weight=sample_weight)
    else:
        scores = []
        n_classes = prob_method.shape[1]
        for i in range(n_classes):
            binary_labels = (y_cal_test == i).astype(int)
            class_probs = prob_method[:, i]
            if len(np.unique(binary_labels)) > 1:
                s = brier_score_loss(binary_labels, class_probs,
                                     sample_weight=sample_weight)
                scores.append(s)
        score = np.mean(scores) if scores else 0.0
    return score