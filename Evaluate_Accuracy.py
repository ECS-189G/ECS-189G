
"""
Concrete evaluator for classification accuracy.
This module computes the overall accuracy score for classification results.
"""
from local_code.base_class.evaluate import evaluate as BaseEvaluator
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(BaseEvaluator):
    """Concrete evaluator for computing classification accuracy."""
    data = None

    def evaluate(self):
        """
        Compute and return the classification accuracy.

        Steps:
        1. Print a start message.
        2. Extract true and predicted labels from the data dictionary.
        3. Calculate the accuracy score using scikit-learn.
        4. Print the computed accuracy with four decimal places.
        5. Return the accuracy value.
        """
        # Notify the beginning of accuracy evaluation
        print('Starting classification accuracy evaluation...')

        # Retrieve labels from the data attribute
        true_label_list = self.data['true_y']
        predicted_label_list = self.data['pred_y']

        # Calculate accuracy score
        accuracy_value = accuracy_score(true_label_list, predicted_label_list)

        # Display the result
        print(f'Classification accuracy: {accuracy_value:.4f}')

        return accuracy_value
