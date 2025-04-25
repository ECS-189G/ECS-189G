"""
Concrete evaluator for multiple classification metrics.
This module computes and returns the accuracy, precision,
recall, and F1 score for classification results.
"""
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    """Evaluates accuracy, precision, recall, and F1 score using full names for clarity."""
    data = None

    def evaluate(self):
        """
        Perform evaluation by computing each metric in full form.

        Prints a start message, then calculates:
        - accuracy_value: overall classification accuracy
        - precision_value: macro-averaged precision
        - recall_value: macro-averaged recall
        - f1_value: macro-averaged F1 score

        Returns:
            str: Formatted string containing each metric result.
        """
        print('Starting evaluation of accuracy, precision, recall, and F1 score...')

        # Retrieve true and predicted labels from data dictionary
        true_labels = self.data['true_y']
        predicted_labels = self.data['pred_y']

        # Compute each metric with descriptive variable names
        accuracy_value = accuracy_score(true_labels, predicted_labels)
        precision_value = precision_score(true_labels, predicted_labels, average='macro')
        recall_value = recall_score(true_labels, predicted_labels, average='macro')
        f1_value = f1_score(true_labels, predicted_labels, average='macro')

        # Format the results into a single string
        results_output = (
            'Accuracy: ' + str(accuracy_value) + '\n'
            + 'Precision: ' + str(precision_value) + '\n'
            + 'Recall: ' + str(recall_value) + '\n'
            + 'F1 Score: ' + str(f1_value)
        )

        return results_output