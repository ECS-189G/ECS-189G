'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import csv
import random
from collections import Counter
from local_code.base_class.dataset import dataset
from nltk.tokenize import word_tokenize
import torch # Added torch import for tensor creation/handling

class JokeDataLoader(dataset): # Inherits from local_code.base_class.dataset
    data = None # Will store tokenized list of all words
    encoded_data = None # Will store encoded (integer ID) list of all words
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_to_int = None
    int_to_vocab = None
    vocab_size = None

    # Define special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    END_OF_JOKE_TOKEN = 'END_OF_JOKE' # Used in JokeRNNMethod

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        # Add dataset_name attribute explicitly for clarity and compatibility with Joke_Setting
        self.dataset_name = dName if dName is not None else "Unnamed Joke Dataset"

        # Initialize instance attributes for data loading
        self.dataset_source_folder_path = None
        self.dataset_source_file_name = None

        # Initialize vocab/encoded data to None, they will be populated by load()
        self.data = None
        self.encoded_data = None
        self.vocab_to_int = None
        self.int_to_vocab = None
        self.vocab_size = None

        # Ensure end_of_joke_token is an instance attribute for JokeRNNMethod to find
        self.end_of_joke_token = JokeDataLoader.END_OF_JOKE_TOKEN


    def set_vocab(self, all_data_tokens):
        """
        Builds vocabulary and creates mappings from words to integers and vice versa.
        Args:
            all_data_tokens (list): A flattened list of all tokens from the dataset.
        """
        word_count = Counter(all_data_tokens)

        # Sort words by frequency (excluding special tokens for sorting)
        # and then add special tokens at the beginning with fixed IDs.
        sorted_unique_words = sorted([word for word in word_count if word not in [self.PAD_TOKEN, self.UNK_TOKEN, self.END_OF_JOKE_TOKEN]],
                                     key=word_count.get, reverse=True)

        # Assign fixed IDs to special tokens
        self.vocab_to_int = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.END_OF_JOKE_TOKEN: 2
        }
        current_idx = 3 # Start IDs for regular words after special tokens

        for word in sorted_unique_words:
            if word not in self.vocab_to_int: # Should always be true for sorted_unique_words
                self.vocab_to_int[word] = current_idx
                current_idx += 1

        self.int_to_vocab = {idx: word for word, idx in self.vocab_to_int.items()}
        self.vocab_size = len(self.vocab_to_int)
        print(f"[{self.dataset_name}] Vocabulary size: {self.vocab_size}")

    def preprocess_jokes(self, raw_data):
        """
        Tokenizes raw joke strings, adds END_OF_JOKE token, shuffles,
        flattens, builds vocabulary, and encodes data into integer IDs.
        Args:
            raw_data (list): A list of raw joke strings.
        """
        tokenized_data = []
        for joke in raw_data:
            tokens = word_tokenize(joke.lower())
            tokens.append(self.END_OF_JOKE_TOKEN) # Add end of joke token
            tokenized_data.append(tokens)

        # Shuffle jokes (shuffling entire joke lists before flattening helps mix contexts)
        random.seed(42) # Use a seed for reproducibility
        random.shuffle(tokenized_data)

        # Flatten the list of lists of tokens into a single list of tokens
        tokenized_data_flat = [token for joke in tokenized_data for token in joke]
        self.data = tokenized_data_flat # Store the flattened token list

        # Set index lookup tables for tokens in dataset vocabulary
        self.set_vocab(tokenized_data_flat)

        # Encode using lookup table
        self.encoded_data = [self.vocab_to_int.get(token, self.vocab_to_int[self.UNK_TOKEN]) for token in tokenized_data_flat]

        print(f"[{self.dataset_name}] Preprocessing complete. Total tokens: {len(self.encoded_data)}")


    def preprocess_text(self, text_string: str) -> list[str]:
        """
        Preprocesses a single string for inference/generation (e.g., initial prompt).
        Args:
            text_string (str): The input text string.
        Returns:
            list[str]: A list of preprocessed tokens.
        """
        # Ensure NLTK punkt tokenizer is downloaded: `nltk.download('punkt')`
        # This will be called by JokeRNNMethod.generate_text
        return word_tokenize(text_string.lower())

    def encode_text(self, text_tokens: list[str]) -> list[int]:
        """
        Encodes a list of text tokens into integer IDs using the learned vocabulary.
        Args:
            text_tokens (list[str]): A list of string tokens.
        Returns:
            list[int]: A list of integer IDs.
        """
        return [self.vocab_to_int.get(token, self.vocab_to_int[self.UNK_TOKEN]) for token in text_tokens]

    def decode_text(self, encodings: list[int]) -> list[str]:
        """
        Decodes a list of integer IDs back into text tokens.
        Args:
            encodings (list[int]): A list of integer IDs.
        Returns:
            list[str]: A list of string tokens.
        """
        return [self.int_to_vocab.get(encoding, self.UNK_TOKEN) for encoding in encodings]


    def load(self):
        """
        Loads raw joke data from a CSV file, then preprocesses it.
        The CSV is assumed to have a header, and jokes are in the second column (index 1).
        """
        full_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        print(f"[{self.dataset_name}] Loading data from {full_path}...")

        raw_jokes = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # Skip header row
                for line in reader:
                    if len(line) > 1: # Ensure there's a second column
                        raw_jokes.append(line[1])
            print(f"[{self.dataset_name}] Loaded {len(raw_jokes)} raw jokes.")
            self.preprocess_jokes(raw_jokes)
            print(f"[{self.dataset_name}] Data loaded and preprocessed.")
            return self.encoded_data # Return the encoded data as expected by JokeRNNMethod
        except FileNotFoundError:
            print(f"[{self.dataset_name}] Error: Data file not found at {full_path}. Please check the path and filename.")
            raise # Re-raise to stop execution
        except Exception as e:
            print(f"[{self.dataset_name}] An error occurred during data loading or preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise # Re-raise to stop execution