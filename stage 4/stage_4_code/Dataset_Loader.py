from collections import Counter
import numpy as np
import json
import torch

class Dataset_Loader:
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription

    def load(self):
        print('loading data...')

        print("loading from ", self.dataset_source_folder_path + self.dataset_source_file_name)
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        glove_embeddings = load_glove_embeddings('../../data/stage_4_data/glove.6B/glove.6B.100d.txt')
        glove_embed_size = 100

        vocab_set = set()
        for sentiment in ['pos', 'neg']:
            for review in data['train'][sentiment]:
                vocab_set.update(review)
            for review in data['test'][sentiment]:
                vocab_set.update(review)

        counter = Counter(vocab_set)
        vocab = sorted(counter, key=counter.get, reverse=True)
        int2word = dict(enumerate(vocab, 1))
        int2word[0] = '<PAD>'
        word2int = {word: id for id, word in int2word.items()}

        X_train, X_test, y_train, y_test = [], [], [], []

        for sentiment in ['pos', 'neg']:
            for review in data['train'][sentiment]:
                review_enc = [word2int[word] for word in review]
                X_train.append(review_enc)
                y_train.append(1 if sentiment == 'pos' else 0)
            for review in data['test'][sentiment]:
                review_enc = [word2int[word] for word in review]
                X_test.append(review_enc)
                y_test.append(1 if sentiment == 'pos' else 0)

        seq_length = 256
        X_train_pad = pad_features(X_train, pad_id=word2int['<PAD>'], seq_length=seq_length)
        X_test_pad = pad_features(X_test, pad_id=word2int['<PAD>'], seq_length=seq_length)

        embedding_matrix = torch.zeros((len(vocab_set) + 1, glove_embed_size))
        for word, i in word2int.items():
            if word in glove_embeddings:
                embedding_matrix[i] = glove_embeddings[word]
            else:
                embedding_matrix[i] = 0

        print("size of vocab", len(vocab_set))
        print("size of glove", len(embedding_matrix))

        #
        return {
            'X_train': torch.tensor(X_train_pad, dtype=torch.long),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test': torch.tensor(X_test_pad, dtype=torch.long),
            'y_test': torch.tensor(y_test, dtype=torch.long),
            'embedding': embedding_matrix
        }

def pad_features(reviews, pad_id, seq_length=128):
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)
    for i, row in enumerate(reviews):
        features[i, :len(row)] = np.array(row)[:seq_length]
    return features

def load_glove_embeddings(glove_file):
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings_dict[word] = vector
    return embeddings_dict
