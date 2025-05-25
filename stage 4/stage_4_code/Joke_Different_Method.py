import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import traceback

# --- Import your actual local classes ---
try:
    from local_code.stage_4_code.Joke_Dataset_Loader import JokeDataLoader
    from local_code.stage_4_code.Joke_Result_Saver import Result_Saver as JokeResultSaver
    from local_code.stage_4_code.Joke_Setting import Setting
    from local_code.stage_4_code.Joke_Evaluate_Metrics import Evaluate_Metrics as JokeEvaluateMetrics
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(
        "Please ensure your 'local_code' directory is correctly configured in PYTHONPATH or is in the same directory.")
    exit()


# --- Placeholder for 'method' base class ---
class method:
    def __init__(self, mName, mDescription):
        self.method_name = mName
        self.method_description = mDescription

    def run(self, *args, **kwargs):
        raise NotImplementedError


# --- Placeholder for 'JokeDataset' ---
class JokeDataset(Dataset):
    def __init__(self, encoded_data, seq_len):
        if not isinstance(encoded_data, list):
            raise TypeError("encoded_data must be a list of integer IDs.")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty for JokeDataset.")

        if len(encoded_data) <= seq_len:
            raise ValueError(f"Length of encoded_data ({len(encoded_data)}) in JokeDataset "
                             f"must be greater than seq_len ({seq_len}) to form at least one sequence-target pair.")

        self.encoded_data = encoded_data
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []

        for i in range(len(encoded_data) - seq_len):
            self.sequences.append(encoded_data[i:i + seq_len])
            self.targets.append(encoded_data[i + seq_len])

        if not self.sequences:
            raise ValueError("No sequences could be created in JokeDataset. "
                             "Check data length and seq_len again.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


# --- Main JokeRNNMethod Class ---
class JokeRNNMethod(nn.Module, method):
    DEFAULT_END_OF_JOKE_TOKEN = "END_OF_JOKE"  # Fallback if dataloader doesn't provide
    model_save_path_template = "./{rnn_type}_joke-weights.pth"
    plot_save_path_template = "./{rnn_type}_joke_loss.png"

    def __init__(self, mName, mDescription, dataloader_obj,  # dataloader_obj is now fully loaded
                 rnn_type='lstm',
                 embedding_size=200,
                 rnn_size=500,
                 num_layers=3,
                 dropout_rate=0.2,
                 learning_rate=1e-3,
                 max_epochs=50,
                 batch_size=128,
                 seq_len=3,
                 temperature=1.0,
                 device=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"JokeRNNMethod '{mName}' using device: {self.device}")

        # --- Data Loader Validation and Assignment ---
        # dataloader_obj is now ASSUMED to be already loaded when passed to constructor.
        # So, no need to call dataloader_obj.load() here.

        # Check required attributes for the already loaded dataloader_obj.
        required_attrs = ['vocab_size', 'encoded_data', 'vocab_to_int', 'int_to_vocab']
        for attr in required_attrs:
            if not hasattr(dataloader_obj, attr) or getattr(dataloader_obj, attr) is None:
                raise AttributeError(f"dataloader_obj is missing or has None for required attribute: '{attr}'. "
                                     f"Ensure {type(dataloader_obj).__name__}.load() is called and successfully populates these attributes BEFORE passing to JokeRNNMethod.")

        self.dataloader_obj = dataloader_obj
        self.vocab_size = self.dataloader_obj.vocab_size  # Assign vocab_size to self for model layers

        # --- Model Hyperparameters ---
        self.rnn_type = rnn_type.lower()
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.temperature = temperature

        self.model_save_path = self.model_save_path_template.format(rnn_type=self.rnn_type)
        self.plot_save_path = self.plot_save_path_template.format(rnn_type=self.rnn_type)

        self._initialize_layers()  # Now self.vocab_size is guaranteed to be an integer
        self.to(self.device)

    def _initialize_layers(self):
        print(f"Initializing layers for {self.method_name}: RNN type: {self.rnn_type.upper()}, "
              f"vocab_size: {self.vocab_size}, embedding_size: {self.embedding_size}, "
              f"rnn_size: {self.rnn_size}, num_layers: {self.num_layers}")
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        rnn_layer_dropout_param = self.dropout_rate if self.num_layers > 1 else 0
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size, self.rnn_size, self.num_layers,
                              dropout=rnn_layer_dropout_param, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, self.rnn_size, self.num_layers,
                               dropout=rnn_layer_dropout_param, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: '{self.rnn_type}'. Choose 'gru' or 'lstm'.")

        self.fc = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        final_out = self.fc(out[:, -1, :])
        return final_out, hidden

    def init_hidden(self, current_batch_size):
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.num_layers, current_batch_size, self.rnn_size, device=self.device),
                    torch.zeros(self.num_layers, current_batch_size, self.rnn_size, device=self.device))
        else:
            return torch.zeros(self.num_layers, current_batch_size, self.rnn_size, device=self.device)

    def train_model(self):
        print(f"Starting model training for {self.method_name} ({self.rnn_type.upper()})...")
        self.train()

        epoch_losses = []
        try:
            dataset = JokeDataset(self.dataloader_obj.encoded_data, self.seq_len)
        except ValueError as e:
            print(f"Error creating JokeDataset: {e}. Cannot train.")
            return

        if len(dataset) == 0:
            print("JokeDataset is empty. Cannot train.")
            return

        effective_batch_size = min(self.batch_size, len(dataset))
        if effective_batch_size == 0:
            print("Effective batch size is 0. Cannot train.")
            return

        batch_dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=True)
        if len(batch_dataloader) == 0:
            print(
                f"DataLoader is empty (dataset size {len(dataset)}, batch_size {effective_batch_size}, drop_last=True). "
                "Trying without drop_last or with smaller batch size if possible.")
            batch_dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
            if len(batch_dataloader) == 0:
                print("DataLoader is still empty even with drop_last=False. Cannot train.")
                return

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        best_loss = float('inf')
        best_model_state = None

        print(
            f"Training for {self.max_epochs} epochs with batch size {effective_batch_size} and {len(batch_dataloader)} batches per epoch.")

        for epoch in range(self.max_epochs):
            self.train()
            epoch_total_loss = 0
            num_valid_batches = 0

            for batch_idx, (X_batch, y_batch) in enumerate(batch_dataloader):
                current_actual_batch_size = X_batch.size(0)
                if current_actual_batch_size == 0: continue

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred, _ = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()

                epoch_total_loss += loss.item()
                num_valid_batches += 1

                if batch_idx > 0 and batch_idx % 50 == 0:
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Batch {batch_idx}/{len(batch_dataloader)}, "
                          f"Batch Loss: {loss.item():.4f}")

            if num_valid_batches == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs} had no valid batches. Skipping loss calculation.")
                continue

            avg_epoch_loss = epoch_total_loss / num_valid_batches
            epoch_losses.append(avg_epoch_loss)
            print(f'Epoch {epoch + 1}/{self.max_epochs}, Average Training Loss: {avg_epoch_loss:.4f}')

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = self.state_dict().copy()
                print(f"New best model state found with loss: {best_loss:.4f}")

        if best_model_state:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            save_dict = {
                'model_state_dict': best_model_state,
                'vocab_to_int': self.dataloader_obj.vocab_to_int,
                'int_to_vocab': self.dataloader_obj.int_to_vocab,
                'rnn_type': self.rnn_type,
                'rnn_size': self.rnn_size,
                'embedding_size': self.embedding_size,
                'num_layers': self.num_layers,
                'seq_len': self.seq_len,
                'vocab_size': self.vocab_size,
                'dropout_rate': self.dropout_rate
            }
            torch.save(save_dict, self.model_save_path)
            print(f"Best model state for {self.rnn_type.upper()} saved to {self.model_save_path}")
        else:
            print("No improvement in model found during training. Model not saved.")

        if epoch_losses:
            os.makedirs(os.path.dirname(self.plot_save_path), exist_ok=True)
            plt.figure()
            plt.plot(epoch_losses)
            plt.xlabel('Epochs')
            plt.ylabel('Average Loss')
            plt.title(f'{self.rnn_type.upper()} Training Loss over Epochs')
            plt.savefig(self.plot_save_path)
            plt.close()
            print(f"Training loss plot for {self.rnn_type.upper()} saved to {self.plot_save_path}")
        else:
            print("No losses recorded, skipping plot generation.")

    def generate_text(self, start_tokens, max_tokens=50):
        cli_start_tokens_str = ' '.join(start_tokens if isinstance(start_tokens, list) else str(start_tokens).split())
        print(
            f"\nGenerating text for {self.method_name} ({self.rnn_type.upper()}) starting with: '{cli_start_tokens_str}'")
        self.eval()

        local_vocab_to_int = None
        local_int_to_vocab = None
        generation_seq_len = self.seq_len

        try:
            if not os.path.exists(self.model_save_path):
                raise FileNotFoundError(f"Model file not found at {self.model_save_path}")

            checkpoint_data = torch.load(self.model_save_path, map_location=self.device)

            if isinstance(checkpoint_data, dict):
                print(f"Loading model from new dictionary format checkpoint ({self.model_save_path}).")

                loaded_rnn_type = checkpoint_data.get('rnn_type', self.rnn_type)
                loaded_vocab_size_ckpt = checkpoint_data.get('vocab_size')
                if loaded_vocab_size_ckpt is not None and self.vocab_size != loaded_vocab_size_ckpt:
                    print(f"CRITICAL WARNING: Current dataloader vocab size ({self.vocab_size}) "
                          f"differs from checkpoint vocab size ({loaded_vocab_size_ckpt}). This WILL cause issues if not handled by re-initializing layers.")

                loaded_embedding_size = checkpoint_data.get('embedding_size', self.embedding_size)
                loaded_rnn_size = checkpoint_data.get('rnn_size', self.rnn_size)
                loaded_num_layers = checkpoint_data.get('num_layers', self.num_layers)
                generation_seq_len = checkpoint_data.get('seq_len', self.seq_len)

                if (self.rnn_type != loaded_rnn_type or
                        self.embedding_size != loaded_embedding_size or
                        self.rnn_size != loaded_rnn_size or
                        self.num_layers != loaded_num_layers):
                    print(
                        "Warning: Model architecture parameters in checkpoint differ from current instance configuration.")
                    print("Updating instance parameters and re-initializing layers to match checkpoint for loading.")
                    self.rnn_type = loaded_rnn_type
                    self.embedding_size = loaded_embedding_size
                    self.rnn_size = loaded_rnn_size
                    self.num_layers = loaded_num_layers
                    self._initialize_layers()
                    self.to(self.device)

                self.load_state_dict(checkpoint_data['model_state_dict'])
                local_vocab_to_int = checkpoint_data['vocab_to_int']
                local_int_to_vocab = checkpoint_data.get('int_to_vocab')

            elif isinstance(checkpoint_data, list):
                print(f"Loading model from old list format checkpoint ({self.model_save_path}).")
                if len(checkpoint_data) == 2:
                    self.load_state_dict(checkpoint_data[0])
                    local_vocab_to_int = checkpoint_data[1]
                    print("Warning: Model architectural parameters (rnn_type, rnn_size, etc.) "
                          "are not in old checkpoint. Ensure current model instance was initialized correctly to match saved state.")
                else:
                    raise ValueError("Old checkpoint list format is unrecognized (expected 2 elements).")
            else:
                raise TypeError(f"Unrecognized checkpoint format: {type(checkpoint_data)}")

            if local_int_to_vocab is None and local_vocab_to_int is not None:
                print("Reconstructing int_to_vocab from vocab_to_int.")
                local_int_to_vocab = {i: token for token, i in local_vocab_to_int.items()}

            if local_vocab_to_int is None or local_int_to_vocab is None:
                raise ValueError("Vocabulary (vocab_to_int or int_to_vocab) could not be loaded/reconstructed.")

            print("Model loaded successfully for generation.")

        except FileNotFoundError:
            print(f"Error: Model file '{self.model_save_path}' not found. Training may be required.")
            return f"Error: Model not found at {self.model_save_path}."
        except Exception as e:
            print(f"Error during model loading or vocabulary setup: {e}")
            traceback.print_exc()
            return f"Error loading model: {e}"

        eoj_token_string_to_use = getattr(self.dataloader_obj, 'end_of_joke_token', self.DEFAULT_END_OF_JOKE_TOKEN)
        if not hasattr(self.dataloader_obj, 'end_of_joke_token'):
            print(f"Warning: '{type(self.dataloader_obj).__name__}' object has no 'end_of_joke_token' attribute. "
                  f"Using default stop token: '{self.DEFAULT_END_OF_JOKE_TOKEN}'. Please ensure your DataloaderLike class has it.")
        if not isinstance(eoj_token_string_to_use, str):
            print(f"Warning: end-of-joke token string is not valid ('{eoj_token_string_to_use}'). "
                  f"Falling back to class default '{self.DEFAULT_END_OF_JOKE_TOKEN}'.")
            eoj_token_string_to_use = self.DEFAULT_END_OF_JOKE_TOKEN

        if isinstance(start_tokens, str):
            if hasattr(self.dataloader_obj, 'preprocess_text') and callable(self.dataloader_obj.preprocess_text):
                processed_start_tokens = self.dataloader_obj.preprocess_text(start_tokens)
            else:
                print("Warning: dataloader_obj has no preprocess_text method. Using basic string.lower().split().")
                processed_start_tokens = start_tokens.lower().split()
        elif isinstance(start_tokens, list) and all(isinstance(t, str) for t in start_tokens):
            processed_start_tokens = start_tokens
        else:
            err_msg = "Error: start_tokens must be a string or a list of strings."
            print(err_msg)
            return err_msg

        current_sequence_ids = [local_vocab_to_int.get(token, local_vocab_to_int.get('<UNK>', 0))
                                for token in processed_start_tokens]
        if not current_sequence_ids and processed_start_tokens:
            print("Warning: start_tokens resulted in empty sequence_ids (all unknown?). Generation might be poor.")

        generated_ids = list(current_sequence_ids)
        hidden = self.init_hidden(1)

        for _ in range(max_tokens):
            if not current_sequence_ids:
                print("Stopping generation: current input sequence for model is empty.")
                break

            if len(current_sequence_ids) < generation_seq_len:
                input_ids_for_model = current_sequence_ids
            else:
                input_ids_for_model = current_sequence_ids[-generation_seq_len:]

            if not input_ids_for_model:
                print("Stopping generation: input_ids_for_model became empty.")
                break

            input_tensor = torch.tensor([input_ids_for_model], dtype=torch.long).to(self.device)

            with torch.no_grad():
                output_logits, hidden = self(input_tensor, hidden)

            probabilities = torch.softmax(output_logits.squeeze(0) / self.temperature, dim=0)
            next_token_id = torch.multinomial(probabilities, 1).item()
            next_token_as_string = local_int_to_vocab.get(next_token_id)

            if next_token_as_string == eoj_token_string_to_use:
                print(f"Generated end-of-joke token ('{eoj_token_string_to_use}'). Stopping generation.")
                break

            generated_ids.append(next_token_id)
            current_sequence_ids.append(next_token_id)

        if generated_ids and local_int_to_vocab.get(generated_ids[-1]) == eoj_token_string_to_use:
            generated_ids.pop()

        decoded_text_list = [local_int_to_vocab.get(idx, '<UNK>') for idx in generated_ids]
        return ' '.join(decoded_text_list)

    def run(self, start_tokens_for_generation, training_mode=False):
        print(f"Method '{self.method_name}' ({self.rnn_type.upper()}) running...")
        if training_mode:
            print("--Start training--")
            self.train_model()
        else:
            print("--Skipping training, proceeding to generation--")

        generated_text_result = self.generate_text(start_tokens_for_generation)
        return generated_text_result


# --- Example Main Script Section (illustrative, adapt to your project structure) ---
if __name__ == '__main__':
    # --- Script-level Configurations ---
    SCRIPT_SEED = 42  # Changed seed for potentially different outcome from previous example runs
    # This controls whether the 'run' method triggers training or just generation
    SCRIPT_TRAINING_MODE = True  # Set to True to train, False to only generate (requires pre-trained model)
    SCRIPT_START_PROMPT = "knock knock"  # A classic joke starter

    # Choose RNN Type for this specific experiment: 'gru' or 'lstm'
    TARGET_RNN_TYPE = 'lstm'  # <<--- CHANGE THIS TO 'lstm' TO RUN LSTM EXPERIMENT

    # Define paths (consider making these absolute or more robust if script is moved)
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..') # Adjust as needed
    # MAIN_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'stage_4_data', 'text_generation')
    # MAIN_RESULT_FOLDER_PATH = os.path.join(PROJECT_ROOT_DIR, 'result', 'stage_4_result')
    MAIN_DATA_FOLDER_PATH = r'../../data/stage_4_data/text_generation/'  # Relative paths
    MAIN_RESULT_FOLDER_PATH = r'../../result/stage_4_result/'

    MAIN_DATASET_SOURCE_FILE_NAME = 'data'  # Placeholder for your data file/identifier
    # Result filename will include RNN type from JokeRNNMethod's model_save_path template
    # The Result_Saver will use the path configured in method_obj by default if not set differently.
    # Let's make result filename also dynamic for clarity in this example.
    MAIN_RESULT_DESTINATION_FILE_NAME = f'{TARGET_RNN_TYPE}_generated_jokes_output.txt'


    # --- Dummy Dataloader for testing this script structure ---
    class DataloaderLike:
        def __init__(self, num_samples=5000, vocab_sz=60):  # Increased vocab size slightly
            self.vocab_size = vocab_sz
            self.unk_token = '<UNK>'
            self.pad_token = '<PAD>'
            self.end_of_joke_token = "END_OF_JOKE"  # Crucial: JokeRNNMethod uses this as default if not on dataloader
            # It's good practice for the dataloader to define it.

            self.vocab_to_int = {f"word_{i}": i for i in range(self.vocab_size - 3)}
            self.vocab_to_int[self.unk_token] = self.vocab_size - 3
            self.vocab_to_int[self.pad_token] = self.vocab_size - 2
            self.vocab_to_int[self.end_of_joke_token] = self.vocab_size - 1
            self.int_to_vocab = {i: token for token, i in self.vocab_to_int.items()}

            # Generate more varied dummy data
            base_words = [self.vocab_to_int[f"word_{i}"] for i in range(self.vocab_size - 3)]
            self.encoded_data = []
            for _ in range(num_samples // 10):  # Create ~num_samples/10 sentences
                sent_len = np.random.randint(5, 15)
                sentence = np.random.choice(base_words, size=sent_len).tolist()
                self.encoded_data.extend(sentence)
                if np.random.rand() < 0.7:  # 70% chance to end with EOJ
                    self.encoded_data.append(self.vocab_to_int[self.end_of_joke_token])
            if not self.encoded_data:  # Ensure it's not empty
                self.encoded_data = np.random.randint(0, self.vocab_size - 3, 100).tolist()

        def load(self):
            print(f"DataloaderLike: Data loaded/prepared (vocab size: {self.vocab_size}).")

        def preprocess_text(self, text_string: str):
            return text_string.lower().split()


    # --- End DataloaderLike ---

    def set_script_seeds(seed_value):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        print(f"Script-level random seeds set to: {seed_value}")


    def script_main():
        set_script_seeds(SCRIPT_SEED)
        print(f"{'*' * 12} Main Script Start {'*' * 12}")

        print("Initializing objects for the script...")
        try:
            # Replace DataloaderLike with your actual JokeDataLoader instantiation
            dataloader_for_run = DataloaderLike(num_samples=20000, vocab_sz=100)  # More data for dummy

            # --- Configure JokeRNNMethod Instance ---
            # These are the parameters you'd vary for Stage 4-5 experiments
            method_params = {
                'rnn_type': TARGET_RNN_TYPE,
                'embedding_size': 256,
                'rnn_size': 512,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'max_epochs': 20,  # Adjust for quicker/longer training during experiments
                'batch_size': 64,
                'seq_len': 5,  # Experiment with different context windows
                'temperature': 0.9  # Experiment with temperature
            }
            print(f"Instantiating JokeRNNMethod with params: {method_params}")

            method_for_run = JokeRNNMethod(
                mName=f'{TARGET_RNN_TYPE.upper()} Joke Experiment',
                mDescription=f'Experimental {TARGET_RNN_TYPE.upper()} model for joke generation.',
                dataloader_obj=dataloader_for_run,
                **method_params  # Pass all defined hyperparameters
            )

            result_saver_for_run = JokeResultSaver('Experiment Result Saver', 'Saves experimental joke outputs.')
            result_saver_for_run.result_destination_folder_path = MAIN_RESULT_FOLDER_PATH
            # Result filename now uses the rnn_type from the method object and a clear name
            result_saver_for_run.result_destination_file_name = f"{method_for_run.rnn_type}_generated_jokes_output.txt"

            setting_orchestrator = Setting('Experiment Setting', 'Orchestrates experimental runs.')
            evaluator_for_run = JokeEvaluateMetrics('Experiment Evaluator', 'Evaluates generated jokes.')

            print("Objects initialized.")
        except Exception as e:
            print(f"Error during object initialization in script_main: {e}")
            traceback.print_exc()
            return

        # ---- Running Section (using the Setting object to orchestrate) ----
        try:
            print("\nPreparing setup via Setting object...")
            setting_orchestrator.prepare(dataloader_for_run, method_for_run, result_saver_for_run, evaluator_for_run)
            print("\nSetup Summary (via Setting object):")
            setting_orchestrator.print_setup_summary()

            print(f"\nInvoking run via Setting object (Training Mode: {SCRIPT_TRAINING_MODE})...")
            generated_text_output = setting_orchestrator.load_run_save_evaluate(
                start_tokens=SCRIPT_START_PROMPT,
                training=SCRIPT_TRAINING_MODE  # This flag determines if model.train_model() is called
            )

            print(f"\n{'*' * 12} Final Generated Text from Script ({TARGET_RNN_TYPE.upper()}) {'*' * 12}")
            if isinstance(generated_text_output, str) and generated_text_output.startswith("Error:"):
                print(f"Main script: Generation process reported an error: {generated_text_output}")
            elif generated_text_output and isinstance(generated_text_output, str):
                print(generated_text_output)
            else:
                print("Main script: No text was generated, or an unexpected result type was returned.")

        except Exception as e:
            print(f"An error occurred during the main script's running section: {e}")
            traceback.print_exc()
        finally:
            print(f"\n{'*' * 12} Main Script Finish {'*' * 12}")


    script_main()