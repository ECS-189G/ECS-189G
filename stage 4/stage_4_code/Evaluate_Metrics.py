# local_code/stage_4_code/Evaluate_Metrics.py

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Evaluate_Metrics(evaluate):
    def __init__(self, name, description):
        super().__init__(name, description)
        self.data = None

    def evaluate(self, data):
        """
        data: dict with keys 'true_y' and 'pred_y', each a sequence of labels
        """
        self.data = data
        print('evaluating performance...')

        acc   = accuracy_score(self.data['true_y'], self.data['pred_y'])
        prec  = precision_score(self.data['true_y'], self.data['pred_y'], average='macro')
        rec   = recall_score(self.data['true_y'], self.data['pred_y'], average='macro')
        f1    = f1_score(self.data['true_y'], self.data['pred_y'], average='macro')

        # Return a formatted multi‐line string
        return (
            f'Accuracy: {acc}\n'
            f'Precision: {prec}\n'
            f'Recall: {rec}\n'
            f'F1: {f1}'
        )
