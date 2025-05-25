import numpy as np
import torch
import json
import os
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

# Import your local classes. Ensure their paths are correct.
try:
    from local_code.stage_4_code.Result_Saver import Result_Saver
    from local_code.stage_4_code.Setting import Setting
    from local_code.stage_4_code.Method_Classification import Method_Classification
    from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure your 'local_code' directory is correctly configured in PYTHONPATH or is in the same directory.")
    exit()

# --- Dataset_Loader functionality adapted for this script (Manual GloVe Loading) ---
class Dataset_Loader_Adapted:
    def __init__(self, name, description, train_json_path, test_json_path,
                 glove_file_path: str,
                 glove_embedding_dim: int = 100,
                 max_seq_len: int = 500):
        self.name = name
        self.description = description
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.glove_file_path = glove_file_path
        self.glove_embedding_dim = glove_embedding_dim
        self.max_seq_len = max_seq_len

        self.vocab_to_int = None
        self.int_to_vocab = None
        self.glove_embedding_matrix = None
        self.data_dict = {}
        self.vocab_size = 0

    def load(self):
        print(f"[{self.name}] Loading and preparing data from JSON files...")
        try:
            with open(self.train_json_path, 'r', encoding='utf-8') as f:
                train_reviews = json.load(f)
            with open(self.test_json_path, 'r', encoding='utf-8') as f:
                test_reviews = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing cleaned review JSONs. Error: {e}")

        all_reviews = train_reviews['pos'] + train_reviews['neg'] + test_reviews['pos'] + test_reviews['neg']

        print(f"[{self.name}] Building vocabulary...")
        word_counts = defaultdict(int)
        for review_list in all_reviews:
            for word in review_list:
                word_counts[word] += 1

        sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.vocab_to_int = {'<PAD>': 0, '<UNK>': 1}
        current_idx = 2
        for word, _ in sorted_vocab:
            if word not in self.vocab_to_int:
                self.vocab_to_int[word] = current_idx
                current_idx += 1
        self.int_to_vocab = {idx: word for word, idx in self.vocab_to_int.items()}
        self.vocab_size = len(self.vocab_to_int)
        print(f"[{self.name}] Vocabulary size: {self.vocab_size}")

        print(f"[{self.name}] Loading GloVe embeddings from: {self.glove_file_path}...")
        word_to_glove = {}
        glove_loaded_successfully = False
        try:
            with open(self.glove_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    word = parts[0]
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                        if len(vector) == self.glove_embedding_dim:
                            word_to_glove[word] = vector
                    except ValueError:
                        pass

            print(f"[{self.name}] Loaded {len(word_to_glove)} words from GloVe file.")
            if not word_to_glove:
                 raise ValueError("No words loaded from GloVe file. Check file format and path.")

            self.glove_embedding_matrix = torch.zeros(self.vocab_size, self.glove_embedding_dim)
            num_found_words = 0
            for word, idx in self.vocab_to_int.items():
                if word in word_to_glove:
                    self.glove_embedding_matrix[idx] = torch.from_numpy(word_to_glove[word])
                    num_found_words += 1
                else:
                    if word != '<PAD>':
                        self.glove_embedding_matrix[idx] = torch.rand(self.glove_embedding_dim) * 0.5 - 0.25
            print(f"[{self.name}] Populated embedding matrix. Found {num_found_words}/{self.vocab_size} words from GloVe.")
            glove_loaded_successfully = True

        except FileNotFoundError:
            print(f"[{self.name}] CRITICAL ERROR: GloVe file not found at {self.glove_file_path}. Please download it and set the correct path.")
        except Exception as e:
            print(f"[{self.name}] An unexpected error occurred during manual GloVe loading: {e}")
            import traceback
            traceback.print_exc()

        if not glove_loaded_successfully:
            print(f"[{self.name}] Using dummy random embeddings as fallback due to GloVe loading failure.")
            self.glove_embedding_matrix = torch.rand(self.vocab_size, self.glove_embedding_dim) * 2 - 1


        print(f"[{self.name}] Converting reviews to token IDs and padding to {self.max_seq_len}...")
        train_X_ids, train_y = [], []
        for review_list in train_reviews['pos']:
            ids = [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in review_list]
            train_X_ids.append(torch.tensor(ids, dtype=torch.long))
            train_y.append(1)
        for review_list in train_reviews['neg']:
            ids = [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in review_list]
            train_X_ids.append(torch.tensor(ids, dtype=torch.long))
            train_y.append(0)

        test_X_ids, test_y = [], []
        for review_list in test_reviews['pos']:
            ids = [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in review_list]
            test_X_ids.append(torch.tensor(ids, dtype=torch.long))
            test_y.append(1)
        for review_list in test_reviews['neg']:
            ids = [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in review_list]
            test_X_ids.append(torch.tensor(ids, dtype=torch.long))
            test_y.append(0)

        def _pad_and_truncate_sequences(sequences, target_len, pad_value):
            padded_seqs = []
            for seq in sequences:
                if len(seq) > target_len:
                    padded_seqs.append(seq[:target_len])
                else:
                    padded_seqs.append(torch.cat([seq, torch.full((target_len - len(seq),), pad_value, dtype=torch.long)]))
            return torch.stack(padded_seqs)

        train_X_padded = _pad_and_truncate_sequences(train_X_ids, self.max_seq_len, self.vocab_to_int['<PAD>'])
        test_X_padded = _pad_and_truncate_sequences(test_X_ids, self.max_seq_len, self.vocab_to_int['<PAD>'])

        self.data_dict = {
            'train': {'X': train_X_padded, 'y': torch.tensor(train_y, dtype=torch.long)},
            'test': {'X': test_X_padded, 'y': torch.tensor(test_y, dtype=torch.long)},
            'embedding': self.glove_embedding_matrix
        }
        print(f"[{self.name}] Data preparation complete. Train X shape: {self.data_dict['train']['X'].shape}, Test X shape: {self.data_dict['test']['X'].shape}")
        return self.data_dict

# --- Main Script Execution ---
if __name__ == "__main__":
    np_seed = 2
    torch_seed = 2
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    CLEANED_TRAIN_JSON = 'clean_reviews_train.json'
    CLEANED_TEST_JSON = 'clean_reviews_test.json'
    MAX_SEQUENCE_LENGTH = 500

    GLOVE_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'data',
        'stage_4_data',
        'glove.6B',
        'glove.6B.100d.txt'
    )
    GLOVE_DIMENSION = 100

    RESULT_DESTINATION_FOLDER = '../../result/stage_4_result/Text_Classification_'
    RESULT_DESTINATION_FILE_NAME = 'prediction_result.txt'

    print('************ Start Text Classification Pipeline ************')

    try:
        data_loader_obj = Dataset_Loader_Adapted(
            'Classification_Data_Prepper',
            'Loads cleaned JSON, tokenizes, creates vocab, loads GloVe, prepares tensors.',
            train_json_path=CLEANED_TRAIN_JSON,
            test_json_path=CLEANED_TEST_JSON,
            glove_file_path=GLOVE_FILE_PATH,
            glove_embedding_dim=GLOVE_DIMENSION,
            max_seq_len=MAX_SEQUENCE_LENGTH
        )
        prepared_data_for_method = data_loader_obj.load()

        if prepared_data_for_method['embedding'] is None:
             print("CRITICAL: Embedding matrix is None after data loading. Exiting.")
             exit()
        print(f"Quick check on embedding matrix sum (should be non-zero if GloVe loaded): {torch.sum(prepared_data_for_method['embedding'])}")


    except FileNotFoundError as fnfe:
        print(f"Error: Required data file not found. Details: {fnfe}")
        print("Ensure 'clean_reviews_train.json' and 'clean_reviews_test.json' are present.")
        print(f"Also, very importantly, ensure your GLOVE_FILE_PATH is correct: {GLOVE_FILE_PATH}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    embedding_dim_from_glove = prepared_data_for_method['embedding'].shape[1]

    # --- Initialize Method_Classification for Experiment: lstm_frozen_glove ---
    method_obj = Method_Classification(
        mName='text_classification_lstm_frozen_glove', # New name for this experiment
        mDescription='Text classification: BiLSTM, Attention, Complex FC head, FROZEN GloVe.',
        embedding_dim=embedding_dim_from_glove,
        hidden_dim=256,                 # Consistent with complex_head baseline
        num_lstm_layers=2,              # Consistent with complex_head baseline
        batch_size=64,
        learning_rate=0.0005,           # Note: May need tuning when embeddings are frozen
        max_epochs=12,
        grad_clip=3.0,
        dropout_rate=0.4,
        weight_decay=1e-5,
        val_split_percent=0.1,
        patience=5,                     # For early stopping
        seed=torch_seed,
        device=None
    )

    result_obj = Result_Saver('classification_results_saver', 'Saves classification prediction results.')
    result_obj.result_destination_folder_path = RESULT_DESTINATION_FOLDER
    result_obj.result_destination_file_name = RESULT_DESTINATION_FILE_NAME

    setting_obj = Setting('classification_pipeline_setting', 'Orchestrates the text classification pipeline.')
    evaluate_obj = Evaluate_Metrics('classification_metrics_evaluator', 'Calculates standard classification metrics.')

    print('\nPreparing and running the pipeline...')
    print("Running classification...")
    classification_results = method_obj.run(prepared_data_for_method)

    print("\nSaving results...")
    result_obj.save(classification_results)

    print("\nEvaluating metrics...")
    final_metrics = evaluate_obj.evaluate(classification_results)
    print("Evaluation Metrics:", final_metrics)

    print('************ Finish Text Classification Pipeline ************')