'''
    Concrete class for experiment with separate testing and training data
    Does not implement setting base class!
'''
from local_code.base_class.setting import setting


class Setting(setting):
    # testset is typically an instance attribute if it holds mutable data specific to each Setting object
    # For now, it's defined here as a placeholder.
    testset = None

    def __init__(self, mName, mDescription):
        """
        Initializes the Setting object for Joke Generation.
        Args:
            mName (str): The name of the setting.
            mDescription (str): A description of the setting.
        """
        # Call the base class constructor. This is crucial for inheriting attributes like `setting_name`.
        super().__init__(mName, mDescription)

        # Initialize instance-specific attributes
        self.dataset = None
        self.method = None
        self.result = None
        self.evaluate = None
        # self.testset = None # Uncomment and manage if testset is instance-specific

    def prepare(self, sData, sMethod, sResult, sEvaluate):
        """
        Prepares the setting by assigning the data, method, result, and evaluation objects.
        Args:
            sData: The data loader object (e.g., DataloaderLike).
            sMethod: The method object (e.g., JokeRNNMethod).
            sResult: The result saver object (e.g., Joke_Result_Saver).
            sEvaluate: The evaluation metrics object.
        """
        self.dataset = sData
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        print(f"[{self.setting_name}] Setting prepared.")

    def print_setup_summary(self):
        """
        Prints a summary of the current experiment setup.
        Ensures that 'dataset_name' and other required attributes exist on the objects.
        """
        # Safely get dataset_name using getattr in case the dataloader object doesn't have it
        dataset_name = getattr(self.dataset, 'dataset_name',
                               self.dataset.name if hasattr(self.dataset, 'name') else 'Unnamed Dataset')

        print('dataset:', dataset_name,
              ', method:', self.method.method_name,
              ', setting:', self.setting_name,
              # CORRECTED: Use self.result.result_name instead of method_name
              ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)  # Assuming evaluate_metrics has evaluate_name

    def load_run_save_evaluate(self, start_tokens, training):
        """
        Orchestrates the joke generation pipeline: runs the method, saves results, and evaluates (if applicable).
        Args:
            start_tokens (str or list): Initial tokens for text generation.
            training (bool): Flag to indicate if the model should be trained or just generate.
        Returns:
            The generated text result from the method.
        """
        print(f"[{self.setting_name}] Executing load_run_save_evaluate...")

        # Run method (for JokeRNNMethod, dataloader passed at initialization)
        learned_result = self.method.run(start_tokens, training)

        # Save raw result. Joke_Result_Saver expects data to be set to its 'data' attribute,
        # then save() is called without arguments.
        self.result.data = learned_result  # Assign the result to the data attribute of the result saver
        self.result.save()  # Call save without arguments for Joke_Result_Saver

        # Evaluate (if evaluation is meaningful for generated text, e.g., perplexity, or just placeholder)
        # Assuming Joke_Evaluate_Metrics sets its data attribute and its evaluate() method requires no args.
        self.evaluate.data = learned_result  # Assign the result to the data attribute of the evaluator
        evaluation_metrics = self.evaluate.evaluate()  # Call evaluate without arguments
        print(f"[{self.setting_name}] Evaluation complete.")

        return learned_result  # Return the generated text for printing in the main script