import pandas as pd

def possible_overfitting(grid, model_name):

    metrics = 'neg_log_loss'

    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(f'{model_name}_cv_results.tsv', sep='\t', index=True)
    best_idx = grid.best_index_

    train_score = results_df.loc[best_idx, 'mean_train_neg_log_loss']
    val_score = results_df.loc[best_idx, 'mean_test_neg_log_loss']
    score_diff = abs(train_score - val_score)

    print(f'{'=' * 60}')
    print(f'Best model train {metrics} : {-train_score:.4f}')
    print(f'Best model validation {metrics} : {-val_score:.4f}')
    print(f'neg_log_loss gap as indicator of possible overfitting: {score_diff:.4f}')

    train_accuracy = results_df.loc[best_idx, 'mean_train_accuracy']
    val_accuracy = results_df.loc[best_idx, 'mean_test_accuracy']
    accuracy_diff = abs(train_accuracy - val_accuracy)
    print(f'{'-' * 10}')
    print(f'Best model train accuracy: {train_accuracy:.4f}')
    print(f'Best model validation accuracy: {val_accuracy:.4f}')
    print(f'Accuracy gap as indicator of possible overfitting: {accuracy_diff:.4f}')
    print(f'{'=' * 60}')


