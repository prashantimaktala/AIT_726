"""
AIT726 HW 2 Due 3/6/2020
Sentiment classification using Feed Forward Neural Network on dataset of 4181 training and 4182 testing tweets.
Authors: Yasas, Prashanti, Ashwini
Command to run the file: python FeedForward_SentimentClassification.py

Flow:
i. main
ii. run  default parameters: stem = false, binary = true
    1.Train the model
        a. Read the dataset
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
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer as Stemmer
from keras import models
from keras import layers


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
    for label in ['negative', 'positive']:
        tmp = os.path.join(path, label)
        for file in os.listdir(tmp):
            with open(os.path.join(tmp, file), 'r', encoding='utf-8') as f:  # join path to read all the tweets
                tweet = f.read().strip()  # strip function to remove leading and trailing spaces in each tweet
                corpus.append((tweet, 1 if label == 'positive' else 0))
    df = pd.DataFrame.from_records(corpus, columns=['tweet', 'label'])  # build dataframe
    return df


def tokenize(x, stem=False):
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
            tokens.append(t)  # append emoticons without work tokenize
        else:
            tokens += word_tokenize(x)
    if stem:
        tokens = [Stemmer.stem(t) for t in tokens]  # perform stemming
    return tokens


def vocabulary(tokenized_tweets):
    """
    vocabulary - we have created word by word vocabulary for the complete training data for all the provided tokens
    """
    vocab = set()
    for tokens in tokenized_tweets:
        vocab.update(tokens)
    return list(vocab)  # create complete list of tokens for all the tweets


def tf_idf(collection, vocab):
    """
          tf_idf - we have created term frequency - inverse document frequency matrix for all the documents
    """
    tf = [[doc.count(v) for v in vocab] for doc in collection]
    tf = np.array([[1 + np.log(count) if count > 0 else 0 for count in doc] for doc in tf])

    N = len(collection)
    df = [len([1 for doc in collection if w in doc]) for w in vocab]
    idf = np.array([np.log(N / t) for t in df])
    output = tf * idf
    return output


def preprocess(df, stem=False, vocab=None):
    """
    preprocess - calls appropriate tokenize to generate training and test data with required vocabulary
    """
    if stem:
        tokens = df.tweet.apply(lambda x: tokenize(x, True))
    else:
        tokens = df.tweet.apply(tokenize)
    if vocab is None:
        vocab = vocabulary(tokens)

    tfidf = tf_idf(tokens, vocab)

    if not isinstance(tfidf, np.ndarray):
        output = np.array(tfidf.values.tolist())
    return output, vocab, tfidf


def train(features, labels, batch_size=10, learning_rate=0.0001):
    # Set random seed
    np.random.seed(0)
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense( activation='sigmoid', input_shape=(len(features),)))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=20, activation='sigmoid'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=20, activation='sigmoid'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='rms',  # Root Mean Square
                    optimizer='rmsprop',  # Root Mean Square Propagation
                    metrics=['accuracy'])  # Accuracy performance metric

    # Train neural network
    model = network.fit(features,  # Features
                          labels,  # Target vector
                          epochs=3,  # Number of epochs
                          verbose=1,  # Print description after each epoch
                          batch_size=batch_size,)  # Number of observations per batch


def run(stem=False):
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train')
    x_train, vocab, tf = preprocess(df_train, stem=stem)
    y_train = df_train.label.values
    model = train(x_train, y_train)
    df_test = read_files('./data/tweet/test')
    x_test, _ = preprocess(df_test, stem=stem, vocab=vocab)
    y_test = df_test.label.values

def main():
    """
    main - runs all the modules via run function
    """
    logging.info('AIT_726 Feed Forward Neural Network')
    logging.info('Authors: Yasas, Prashanti , Ashwini')
    logging.info(' TFIDF with stemming')
    run(stem=True)

    logging.info('TFIDF without stemming')
    run(stem=False)



if __name__ == '__main__':
    main()