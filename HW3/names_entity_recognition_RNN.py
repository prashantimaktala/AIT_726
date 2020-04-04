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

def sentence_tokenize(x):
    """
    tokenize function takes care of handling removal of conversion of capitalized words to lowercase except
    for all capital words
    """

    x = re.sub('([A-Z][a-z]+)', lambda t: t.group(0).lower(), x)  # group 0 refers to A-Z, lowercase group 0
    tokens = []
    for t in emoticon_tokens:
        if re.match(emoticons_re, t):
            tokens.append(t)  # append emoticons without word tokenize
        else:
            tokens += word_tokenize(x)
    return tokens


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
    for line in open(file_name,"r").readlines():
        if len(line.strip()) == 0 :
            sentences.append(sentence)
            sentence = ""
            labels.append(label.replace("\n", ""))
            label = ""
        else:
            sentence = sentence + (line.split(" ",1)[0]) + " "
            label = label + (line.split(" ")[1]) + " "
            # label = label + (line.split(" ")[3]) +   " "
    return sentences,labels


def preprocess(df):
    """
    Tokenize each tweet and generate positive and negative bigrams.
    Labels are appended to the positive and negative bigrams
    """


    tokens = df.DOCSTART.apply(tokenize)
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
    return x, y


def keras_preprocess(x_train, tokenizer=None, maxlen=2):
    """
    creating bag of words using keras tokenizer
    """
    seperator = ' '
    x_train = [seperator.join(pair) for pair in x_train]

    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    # add padding
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)

    return x_train, tokenizer


def create_model(vocab_size):
    """ create_model -
     - creates feed forward neural network with 2 layers along with embedding layer hidden and vector size 20.
     - initializes the weights with random number.
     - uses mean squared error as our loss function and sigmoid as our activation function
     """
    model = Sequential()
    # set optimizer Adam for the model with learning rate of 0.00001
    # optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # initialize the input layer which contains the embeddings from previous steps
    embedding_layer = Embedding(vocab_size, 100, input_length=2, trainable=False)
    model.add(embedding_layer)
    # flatten the input layer
    model.add(Flatten())
    # The hidden layers with vector size of 20 and activation functon = "sigmoid"
    model.add(Dense(20, activation='sigmoid', kernel_initializer='random_uniform'))
    # the output layer with one output and activation function "sigmoid"
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer=Adam(lr=0.00001),  # Root Mean Square Propagation
                  loss='mse',  # Root Mean Square
                  metrics=['accuracy'])  # Accuracy performance metric
    print(model.summary())
    return model


def validation_train(x_train, y_train, vocab_size):
    """
    validation_train - verifies the accuracy of the model using cross validated training data across
           different hyper parameters. validation_train returns the model with the best accuracy for testing purpose
    """
    best_model = {'accuracy': 0.0, 'model': None, 'hyperparams': {}}
    batch_size = 250
    # Hyper-parameter Search (Grid Search)
    for epochs in [10, 20, 30, 40]:
        for lr in [0.0001, 0.00001]:
            network = create_model(vocab_size)
            # Train neural network
            history = network.fit(x_train,  # Features
                                  y_train,  # Target vector
                                  epochs=epochs,  # Number of epochs
                                  batch_size=batch_size,  # Number of observations per batch
                                  validation_split=0.2)  # Validation split for validation
            if 'val_accuracy' in history.history:
                accuracy = history.history['val_accuracy'][-1]
            elif 'val_acc' in history.history:
                accuracy = history.history['val_acc'][-1]
            else:
                accuracy = 0.0
            if accuracy > best_model['accuracy']:
                best_model['model'] = network
                best_model['accuracy'] = accuracy
                best_model['hyperparams'] = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                }
    model = best_model['model']
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
    df_train = read_files('./Data/conll2003/train.txt')
    sentences, labels = get_sentences('./Data/conll2003/train.txt')

    max_length_sentence = len(max(sentences, key=len))
    pad_tag(df_train, sentences, max_length_sentence)
    x, y = get_sentences_train2('./Data/conll2003/train2.txt')

    word_to_ix = {}
    for sent in x:
        for word in sent.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {}

    for sent in y:
        for word in sent.split():

            if word not in tag_to_ix:
                tag_to_ix[word] = len(tag_to_ix)
    print(tag_to_ix)

    # x_train, y_train = preprocess(df_train)
    # x_train, tokenizer = keras_preprocess(x_train)
    # vocab_size = len(tokenizer.word_index) + 1
    #
    # df_test = read_files('./data/tweet/test/positive')
    # x_test, y_test = preprocess(df_test)
    # x_test, _ = keras_preprocess(x_test, tokenizer)
    #
    # best_model = validation_train(x_train, y_train, vocab_size)
    # best_model.summary()
    # y_pred = best_model.predict(x_test)
    # evaluate(y_test, y_pred.flatten() > 0.5)



def main():
    #test
    """
    main - runs all the modules via run function
    """
    # logging.info('AIT_726 Named Entity Recognition')
    # logging.info('Authors: Yasas, Prashanti , Ashwini')
    run()


if __name__ == '__main__':
    main()