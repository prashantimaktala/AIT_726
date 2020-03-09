"""
AIT726 HW 2 Due 3/6/2020
Sentiment classification using Feed Forward Neural Network on dataset of 4181 training and 4182 testing tweets.
Authors: Yasas, Prashanti, Ashwini
Command to run the file: python FeedForward_SentimentClassification.py

Flow:
i. main
ii. run
        a. Read the  train positive tweets
        b. Perform preprocessing
            - Build vocab
            - tokenize(stem/no stem)
            - extract features
                1.binary bow
                2.freq bow
        c. Train the model
            -
    2. Test the model
        a. Read the dataset
        b. Perform preprocessing
            - tokenize
        c. Predict
        d. Evaluate the model
            - Save confusion matrix and accuracy to log file

"""

import os
import re
import logging
import random
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# use logging to save the results
logging.basicConfig(filename='feedforwardResults.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

# regex from https://stackoverflow.com/questions/28077049/regex-matching-emoticons
emoticons_re = r'(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)'


def read_files(path):
    """
    read_files - helps to navigate through the files in the folder structure, read the files and convert them to
    dataframe which consists of tweet and labels.
    """
    corpus = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:  # join path to read all the tweets
            tweet = f.read().strip()  # strip function to remove leading and trailing spaces in each tweet
            corpus.append(tweet)
    df_new = pd.DataFrame(corpus, columns=['tweet'])  # build dataframe
    return df_new


def tokenize(x):
    """
    tokenize function takes care of handling removal of html tags, conversion of capitalized words to lowercase except
    for all capital words, handling of emoticons. we have created streams of tokens without stemming using word_tokenize
    as well as tokens with stemming using PotterStemmer.
    """
    x = re.sub(r'(?:<[^>]+>)', '', x)  # substitute html tags
    x = re.sub('([A-Z][a-z]+)', lambda t: t.group(0).lower(), x)  # group 0 refers to A-Z, lowercase group 0
    emoticon_tokens = re.split(emoticons_re, x)  # separate emoticons
    tokens = []
    for t in emoticon_tokens:
        if re.match(emoticons_re, t):
            tokens.append(t)  # append emoticons without word tokenize
        else:
            tokens += word_tokenize(x)
    return tokens


def preprocess(df):
    """
    Tokenize each tweet and generate positive and negative bigrams
    """
    tokens = df.tweet.apply(tokenize)
    sent_bigrams = tokens.apply(lambda tweet: [' '.join(bigram) for bigram in ngrams(tweet, 2)])
    x_pos_bigrams = set()
    for sent in sent_bigrams:
        x_pos_bigrams.update(sent)
    x_pos_bigrams = np.array([x.split() for x in x_pos_bigrams])
    vocab = set()
    for token in tokens:
        vocab.update(token)
    vocab = list(vocab)
    # randomly generate two negative samples
    x_neg_bigrams = []
    for _ in range(2):
        for bigrams in x_pos_bigrams:
            t0 = bigrams[0]
            i, t1 = 0, None
            random.shuffle(vocab)
            while t0 == t1 or t1 is None:
                t1 = vocab[i]
                i = i + 1
            x_neg_bigrams.append([t0, t1])
    x_neg_bigrams = np.array(x_neg_bigrams)

    # Add labels for positive and negative samples
    y_pos_bigrams = np.ones(x_pos_bigrams.shape[0])
    y_neg_bigrams = np.zeros(x_neg_bigrams.shape[0])

    x = np.concatenate((x_pos_bigrams, x_neg_bigrams), axis=0)
    y = np.concatenate((y_pos_bigrams, y_neg_bigrams), axis=0)

    # create train and test data using train_test_split()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x, y


def keras_preprocess(x_train, tokenizer=None, maxlen=2):
    # creating bag of words using keras tokenizer
    seperator = ' '
    x_train = [seperator.join(pair) for pair in x_train]

    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    # add padding
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)

    return x_train, tokenizer


# def create_embedding(vocab_size, word_index):
#     embeddings_dictionary = dict()
#     # Using Global Vectors for Word Representation for word embeddings
#     file = open('./glove/glove.6B.100d.txt',
#                       encoding="utf8")
#
#     # We use pretrained glove embeddings i.e the Glove in our case.
#     for line in file:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_dictionary[word] = coefs
#     file.close()
#
#     embedding_matrix = np.zeros((vocab_size, 100))
#     for word, i in word_index.items():
#         embedding_vector = embeddings_dictionary.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding dictionary will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#     print(embedding_matrix)
#     return embedding_matrix


def cross_validate(x_train, y_train, vocab_size):
    seed = 26
    np.random.seed(seed)
    # define 10-fold cross validation test on the training data only
    kfold = KFold(n_splits=10)

    cvscores = []
    for train_index, test_index in kfold.split(x_train):
        # split the data again for kfold validation
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        # create model of FFNN using Sequential()
        model = Sequential()
        # set optimizer Adam for the model with learning rate of 0.00001
        # optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # initialize the input layer which contains the embeddings from previous steps
        embedding_layer = Embedding(vocab_size, 100, input_length=2, trainable=False)
        model.add(embedding_layer)
        # flatten the input layer
        model.add(Flatten())
        # The hidden layers with vector size of 20 and activation functon = "sigmoid"
        model.add(Dense(20, activation='sigmoid'))
        # the output layer with one output and activation function "sigmoid"
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(optimizer=Adam(lr=0.00001),  # Root Mean Square Propagation
                      loss='mse',  # Root Mean Square
                      metrics=['accuracy'])  # Accuracy performance metric
        print(model.summary())
        # Fit the model
        model.fit(X_train,
                  Y_train,
                  epochs=1,
                  batch_size=20,
                  verbose=1)
        # evaluate the model
        scores = model.evaluate(X_test, Y_test, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def train(x_train, y_train, vocab_size):
    model = Sequential()
    # set optimizer Adam for the model with learning rate of 0.00001
    # optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # initialize the input layer which contains the embeddings from previous steps
    embedding_layer = Embedding(vocab_size, 100, input_length=2, trainable=False)
    model.add(embedding_layer)
    # flatten the input layer
    model.add(Flatten())
    # The hidden layers with vector size of 20 and activation functon = "sigmoid"
    model.add(Dense(20, activation='sigmoid'))
    # the output layer with one output and activation function "sigmoid"
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer=Adam(lr=0.00001),  # Root Mean Square Propagation
                  loss='mse',  # Root Mean Square
                  metrics=['accuracy'])  # Accuracy performance metric
    print(model.summary())
    # Fit the model
    model.fit(x_train,
              y_train,
              epochs=1,
              batch_size=20,
              verbose=1)
    return model


def evaluate(y_true, y_pred, true_label=1):
    """
     evaluate - calculates and prints accuracy
     """
    true_positives = sum(np.logical_and(y_true == true_label, y_pred == true_label))
    true_negatives = sum(np.logical_and(y_true != true_label, y_pred != true_label))
    logging.info('Accuracy = %2.2f' % ((true_positives + true_negatives) * 100 / len(y_pred)))
    logging.info('')


def run():
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train/positive')
    x_train, y_train = preprocess(df_train)
    x_train, tokenizer = keras_preprocess(x_train)
    vocab_size = len(tokenizer.word_index) + 1
    # embedding_matrix = create_embedding(vocab_size, word_index)
    cross_validate(x_train, y_train, vocab_size)
    model = train(x_train, y_train, vocab_size)
    df_test = read_files('./data/tweet/test/positive')
    x_test, y_test = preprocess(df_test)
    x_test, _ = keras_preprocess(x_test, tokenizer)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred.flatten())


def main():
    """
    main - runs all the modules via run function
    """
    logging.info('AIT_726 Feed Forward Neural Network')
    logging.info('Authors: Yasas, Prashanti , Ashwini')
    run()


if __name__ == '__main__':
    main()
