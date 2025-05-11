
'''Concrete Evaluate class for multiple evaluation metrics'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        true = self.data['true_y']
        pred = self.data['pred_y']
        # Use zero_division=0 to avoid UndefinedMetricWarning when a label has no predicted samples
        return (
            'Accuracy: ' + str(accuracy_score(true, pred)) + '\n'
            + 'Precision: ' + str(precision_score(true, pred, average='macro', zero_division=0)) + '\n'
            + 'Recall: ' + str(recall_score(true, pred, average='macro', zero_division=0)) + '\n'
            + 'F1: ' + str(f1_score(true, pred, average='macro', zero_division=0))
        )
