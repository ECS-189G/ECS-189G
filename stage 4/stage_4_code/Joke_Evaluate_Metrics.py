'''
Concrete EvaluateModule class for evaluating generated jokes.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate # Assuming this is your base class
# No need for sklearn.metrics for simple text generation evaluation,
# but if you add perplexity or BLEU, you might need other libraries.

class Evaluate_Metrics(evaluate): # Renamed to align with your imports

    data = None # This will hold the generated text string
    evaluate_name = "Joke Evaluation" # Ensure this is set for Setting summary

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)
        self.data = None # Initialize instance data

    def evaluate(self):
        """
        Evaluates the generated text. For joke generation, this is typically
        more qualitative or uses specific metrics like perplexity (not implemented here).
        For this setup, it will just confirm text was generated.
        """
        print(f'[{self.evaluate_name}] Evaluating performance...')

        if self.data is None:
            print(f"[{self.evaluate_name}] No data provided for evaluation.")
            return {'evaluation_message': 'No generated text to evaluate.'}

        if isinstance(self.data, str):
            print(f"[{self.evaluate_name}] Generated text received for evaluation (length: {len(self.data)}).")
            # In a real scenario, you'd calculate metrics here:
            # - Perplexity (requires probability distribution from model)
            # - BLEU/ROUGE (requires reference jokes)
            # - Human evaluation scores
            # For now, it's just a confirmation.
            return {'evaluation_message': 'Text generation evaluation completed (qualitative).',
                    'generated_text_length': len(self.data)}
        else:
            print(f"[{self.evaluate_name}] Unexpected data type for evaluation: {type(self.data)}. Expected string.")
            return {'evaluation_message': 'Unexpected data type for evaluation.',
                    'data_type': str(type(self.data))}