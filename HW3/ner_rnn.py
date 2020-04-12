import argparse
import os
import re
import logging
import random
import gensim
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Embedding, Dense, GRU, SimpleRNN, LSTM, Bidirectional, TimeDistributed, \
    Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import os


def get_sentences(path):
    sentences, labels = [], []
    sentence, label = [], []
    for line in open(path, "r").read().split('\n'):
        if line.startswith('-DOCSTART-'):
            continue
        if len(line.strip()) == 0:
            if len(sentence) == 0:
                continue
            sentences.append(sentence)
            labels.append(label)
            sentence, label = [], []
        else:
            sentence.append(line.split(" ", 1)[0])
            label.append(line.split(" ")[3])
    return sentences, labels


def pad_tag(dataset, max_length_sentence):
    sentences, labels = dataset
    for idx in range(len(sentences)):
        # assert max_length_sentence - len(sentences[idx]) > 0
        sentences[idx] += ['0'] * (max_length_sentence - len(sentences[idx]))
        labels[idx] += ['<pad>'] * (max_length_sentence - len(labels[idx]))
        sentences[idx] = sentences[idx][:max_length_sentence]
        labels[idx] = labels[idx][:max_length_sentence]
    return sentences, labels


def get_vocab(train_sentences):
    vocab = set()
    for sent in train_sentences:
        vocab.update(sent)
    return list(vocab)


def load_word2vec(path='./GoogleNews-vectors-negative300.bin.gz'):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec


def get_embeddings(vocab):
    word2vec = load_word2vec()
    embeddings_index = {'0': 0, '<missing>': 1}
    embeddings = [np.random.random(300) for _ in range(len(vocab) + 2)]
    for word in vocab:
        if (word not in embeddings_index) and (word in word2vec):
            idx = len(embeddings_index)
            embeddings_index[word] = idx
            embeddings[idx] = word2vec[word]
    return np.array(embeddings), embeddings_index


def get_input_seq(sentences, embeddings_index):
    input_seq = []
    for tokens in sentences:
        input_seq.append(
            [embeddings_index[token] if token in embeddings_index else embeddings_index['<missing>'] for token in
             tokens])
    return np.array(input_seq)


def create_rnn_model(embedding_matrix, input_length, rnn='simple_rnn', bidirectional=False, hidden_size=256, lr=0.0001):
    vocabulary_size, embedding_dims = embedding_matrix.shape[0], 300
    model = Sequential()
    model.add(InputLayer(input_shape=(input_length,)))
    model.add(Embedding(vocabulary_size, embedding_dims, weights=[embedding_matrix],
                        input_length=input_length))  # (Batch Size, 50, 300)
    rnn = {
        'simple_rnn': SimpleRNN(hidden_size, return_sequences=True),
        'lstm': LSTM(hidden_size, return_sequences=True),
        'gru': GRU(hidden_size, return_sequences=True),
    }[rnn]
    if bidirectional:
        rnn = Bidirectional(rnn)
    model.add(rnn)
    model.add(TimeDistributed(Dense(10, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    return model


def encode_labels(labels, classes=None):
    if classes is None:
        classes = set()
        for sent in labels:
            classes.update(sent)
        classes = list(classes)
    output = [[[1 if c == lbl else 0 for c in classes] for lbl in sent] for sent in labels]
    return np.array(output), classes


def decode_labels(predictions, classes):
    return np.array([[classes[lbl.argmax()] for lbl in sent] for sent in predictions])


def write_to_file(filename, sentences, labels, preds):
    with open('results_%s.txt' % filename, 'w', encoding='utf-8') as f:
        f.write('Word Gold_Standard Prediction\n')
        for sent_idx, words in enumerate(sentences):
            for word_idx, word in enumerate(words):
                if word != '0' and labels[sent_idx][word_idx] != '<pad>':
                    f.write('%s %s %s\n' % (words[word_idx], labels[sent_idx][word_idx], preds[sent_idx][word_idx]))
            f.write('\n')


def train_validate(model, x_train, y_train, x_valid, y_valid, num_epochs=1, batch_size=2000):
    model.fit(x_train, y_train, epochs=num_epochs, verbose=1, batch_size=batch_size, validation_data=(x_valid, y_valid))
    return model


def main(**kwargs):
    rnn, bidirectional = kwargs.get('rnn', 'simple_rnn'), kwargs.get('bidirectional', False)
    model_name = '%s%s' % ('bi-' if bidirectional else '', rnn)
    input_length = 513
    train_sentences, train_labels = pad_tag(get_sentences('./conll2003/train.txt'), input_length)
    vocab = get_vocab(train_sentences)
    print('Vocab Size: %d' % len(vocab))
    embedding_matrix, embeddings_index = get_embeddings(vocab)
    model = create_rnn_model(
        embedding_matrix,
        input_length,
        rnn=rnn,
        bidirectional=bidirectional,
        hidden_size=256,
        lr=0.0001
    )
    # Train and validate the model
    # -- Convert the input text to (index) sequence
    x_train, (y_train, classes) = get_input_seq(train_sentences, embeddings_index), encode_labels(train_labels)
    # -- Load and Convert the validation text to (index) sequence
    valid_sentences, valid_labels = pad_tag(get_sentences('./conll2003/valid.txt'), input_length)
    x_valid, (y_valid, _) = get_input_seq(valid_sentences, embeddings_index), encode_labels(valid_labels, classes)
    train_validate(model, x_train, y_train, x_valid, y_valid)
    # Save the model
    model.save(model_name + '.h5')
    # Predict for the test data
    test_sentences, test_labels = pad_tag(get_sentences('./conll2003/test.txt'), input_length)
    x_test, (y_test, _) = get_input_seq(test_sentences, embeddings_index), encode_labels(test_labels, classes)
    y_pred = model.predict(x_test)
    write_to_file(model_name, test_sentences, test_labels, decode_labels(y_pred, classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NER Sequence Model.')
    parser.add_argument('--rnn', help='RNN layer type [simple_rnn]/lstm/gru', default='simple_rnn')
    parser.add_argument('--bidirectional', help='whether to use bidirectional true/[false]', default='false')
    args = parser.parse_args()
    main(rnn=args.rnn, bidirectional=args.bidirectional == 'true')
