"""
AIT726 HW 2 Due 4/9/2020
Named entity recognition using RNN
Authors: Yasas, Prashanti, Ashwini
Command to run the file: python names_entity_recognition_RNN.py

Flow:
i. main
ii. run
    1.Train the model
        a. Read the dataset
        b. Perform preprocessing
            - Build vocab
            - tokenize
            - Generate positive bigrams
            - Generate random negative bigrams from postive bigrams
            - Append positive and negative labels for bigrams
        c. Perform keras preprocessing
            - Apply keras tokenizer
        c. create a feed forward neural network ( we have FFNN with 2 layers along with embedding layer and
           hidden vector size 20. We have initialized the weights with random number. We have used mean squared error
           as our loss function and sigmoid as our activation function )
        d. Train the model ( we have verified the accuracy of the model using cross validated training data across
           different hyper parameters. We have later returned the model with the best accuracy for testing purpose )

    2. Test the model
        a. Read the dataset
        b. Perform preprocessing
            - tokenize
        c. Predict ( we have used the model with best accuracy for predicting the test dataset )
        d. Evaluate the model
            - Save accuracy to log file

"""

import os
import re
import logging
import random
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# use logging to save the results
# logging.basicConfig(filename='names_entity_recognition_RNN.log', level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler())

from gensim import models


# embedding_vector = models.KeyedVectors.load_word2vec_format(
#     './Data/conll2003/GoogleNews-vectors-negative300.bin', binary=True)


def read_files(path):
    """
    read_files - helps to navigate through the files in the folder structure, read the files and convert them to
    dataframe which consists of tweet
    """
    df = pd.read_csv(path, sep=" ")
    df = df.astype(str)
    df.drop(df.columns[[1, 2]], axis=1, inplace=True)
    df = df.rename(columns={"-DOCSTART-": "words", "O": "entity"})
    # print(df.head())
    df["words"] = [word.lower() if not word.isupper() else word for word in df["words"]]
    return df


def get_sentences(path):
    sentences = []
    labels = []
    sentence = ""
    label = ""
    for line in open(path, "r").readlines():
        if len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = ""
            labels.append(label.replace("\n", ""))
            label = ""
        else:
            sentence = sentence + (line.split(" ", 1)[0]) + " "
            label = label + (line.split(" ")[3]) + " "

    # max_length_sentence = len(max(sentences, key=len))
    return sentences[1:], labels[1:]


def pad_tag(df, sentences, max_length_sentence):
    start_index = 0
    for sentence in sentences:
        # print(df[start_index:start_index+len(sentence.split())])
        df[start_index:start_index + len(sentence.split())].to_csv(r'./Data/conll2003/train2.txt',
                                                                   header=None, index=None, sep=' ', mode='a')
        file1 = open("./Data/conll2003/train2.txt", "a")
        pad_length = max_length_sentence - len(sentence.split())
        file1.write("0 <pad> \n" * pad_length + ('\n'))

        file1.close()
        start_index = start_index + len(sentence.split())


def get_sentences_train2(file_name):
    sentences = []
    labels = []
    sentence = ""
    label = ""
    for line in open(file_name, "r").readlines():
        if len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = ""
            labels.append(label.replace("\n", ""))
            label = ""
        else:
            sentence = sentence + (line.split(" ", 1)[0]) + " "
            label = label + (line.split(" ")[1]) + " "
            # label = label + (line.split(" ")[3]) +   " "
    return sentences, labels


def get_tag(x, y):
    word_to_ix = {}
    for sent in x:
        for word in sent.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix)
    tag_to_ix = {}

    for sent in y:
        for word in sent.split():

            if word not in tag_to_ix:
                tag_to_ix[word] = len(tag_to_ix)
    return tag_to_ix, word_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def get_label(sequence, tag_to_ix):
    label = []
    # idx2lbl = {y:x for x,y in tag_to_ix.iteritems()}
    idx2lbl = dict([(value, key) for key, value in tag_to_ix.items()])
    for word in sequence:
        label.append(idx2lbl[list(word).index(max(word))])
    return label


def run():
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./Data/conll2003/train.txt')
    sentences, labels = get_sentences('./Data/conll2003/train.txt')

    max_length_sentence = len(max(sentences, key=len))
    pad_tag(df_train, sentences, max_length_sentence)
    x, y = get_sentences_train2('./Data/conll2003/train2.txt')
    # training_data = x, y
    tag_to_ix, word_to_ix = get_tag(x, y)

    print(tag_to_ix)
    print(word_to_ix)

    torch.manual_seed(1)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    training_data = []
    for index, sent in enumerate(x):
        record = [sent.split(), y[index].split()]
        training_data.append(record)
    print(training_data)
    print(len(training_data))


def main():
    # test
    """
    main - runs all the modules via run function
    """
    # logging.info('AIT_726 Named Entity Recognition')
    # logging.info('Authors: Yasas, Prashanti , Ashwini')
    run()


if __name__ == '__main__':
    main()
