"""
AIT726 HW 2 Due 3/6/2020
Sentiment classification using Feed Forward Neural Network on dataset of 4181 training and 4182 testing tweets.
Authors: Yasas, Prashanti, Ashwini
Command to run the file: python FeedForward_SentimentClassification.py

Flow:
i. main
ii. run  default parameters: stem = false
    1.Train the model
        a. Read the dataset
        b. Perform preprocessing
            - Build vocab
            - tokenize(stem/no stem)
            - extracting features by performing tf-idf
        c. create a feed forward neural network ( we have FFNN with 2 layers with hidden vector size 20. We have
           initialized the weights with random number. We have used mean squared error as our loss function and sigmoid
           as our activation function )
        d. Train the model ( we have verified the accuracy of the model using 80:20 split validated training data across
           different hyper parameters. We have later returned the model with the best accuracy for testing purpose )

    2. Test the model
        a. Read the dataset
        b. Perform preprocessing
            - tokenize
        c. Predict ( we have used the model with best accuracy for predicting the test dataset )
        d. Evaluate the model
            - Save confusion matrix and accuracy to log file

"""

import os
import re
import logging
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

stemmer = PorterStemmer()

# use logging to save the results
logging.basicConfig(filename='feedforward_SentimentClassification.log', level=logging.INFO)
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
        tokens = [stemmer.stem(t) for t in tokens]  # perform stemming
    return tokens


def vocabulary(tokenized_tweets):
    """
    vocabulary - we have created word by word vocabulary for the complete training data for all the provided tokens
    """
    vocab = set()
    for tokens in tokenized_tweets:
        vocab.update(tokens)
    return list(vocab)  # create complete list of tokens for all the tweets


def tf_idf(collection, vocab, idf=None):
    """
          tf_idf - we have created term frequency - inverse document frequency matrix for all the documents
    """
    tf = [[doc.count(v) for v in vocab] for doc in collection]
    tf = np.array([[1 + np.log(count) if count > 0 else 0 for count in doc] for doc in tf])
    if idf is None:
        N = len(collection)
        df = [len([1 for doc in collection if w in doc]) for w in vocab]
        idf = np.array([np.log(N / t) for t in df])
    output = tf * idf
    return output, idf


def preprocess(df, stem=False, vocab=None, idf=None):
    """
    preprocess - calls appropriate tokenize to generate training and test data with required vocabulary
    """
    if stem:
        tokens = df.tweet.apply(lambda x: tokenize(x, True))
    else:
        tokens = df.tweet.apply(tokenize)
    if vocab is None:
        vocab = vocabulary(tokens)

    output, idf = tf_idf(tokens, vocab, idf)

    if not isinstance(output, np.ndarray):
        output = np.array(output.values.tolist())

    return output, vocab, idf


def create_model(features, lr):
    """ create_model
     - creates feed forward neural network with 2 layers with hidden vector size 20.
     - initializes the weights with random number.
     - uses mean squared error as our loss function and sigmoid as our activation function
     """
    # Set random seed
    np.random.seed(0)
    # Start neural network
    network = Sequential()

    # Add fully connected layer with a sigmoid activation function
    network.add(Dense(units=20, activation='sigmoid', kernel_initializer='random_uniform', input_dim=features.shape[1]))

    # Add fully connected layer with a sigmoid activation function
    network.add(Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(optimizer=Adam(lr=lr),  # lr = 0.0001
                    loss='mse',  # Root Mean Square
                    metrics=['accuracy'])  # Accuracy performance metric

    return network


def evaluate(y_true, y_pred, true_label=1):
    """ evaluate - calculates and prints accuracy and confusion matrix for predictions
    """
    true_positives = np.sum(np.logical_and(y_true == true_label, y_pred == true_label))
    false_positives = np.sum(np.logical_and(y_true != true_label, y_pred == true_label))
    true_negatives = np.sum(np.logical_and(y_true != true_label, y_pred != true_label))
    false_negatives = np.sum(np.logical_and(y_true == true_label, y_pred != true_label))
    logging.info('Confusion Matrix: ')
    logging.info('\t\tTrue\tFalse')
    logging.info('True\t%d\t\t%d' % (true_positives, false_positives))
    logging.info('False\t%d\t\t%d' % (false_negatives, true_negatives))
    logging.info('Accuracy = %2.2f' % ((true_positives + true_negatives) * 100 / len(y_pred)))
    logging.info('')


def validation_train(x_train, y_train):
    """
    validation_train - verifies the accuracy of the model using 80:20 split validated training data across
           different hyper parameters. validation_train returns the model with the best accuracy for testing purpose
    """
    best_model = {'accuracy': 0.0, 'model': None, 'hyperparams': {}}
    batch_size = 250
    lr = 0.0001
    for epochs in [10, 20, 30, 40]:
        network = create_model(x_train, lr=lr)
        # Train neural network
        history = network.fit(x_train,  # Features
                              y_train,  # Target vector
                              epochs=epochs,  # Number of epochs
                              batch_size=batch_size,  # Number of observations per batch
                              validation_split=0.2)  # Validation split for validation
        accuracy = history.history['val_accuracy'][-1]  # saving the last accuracy value
        if accuracy > best_model['accuracy']:
            best_model['model'] = network
            best_model['accuracy'] = accuracy
            best_model['hyperparams'] = {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
            }
    logging.info('Best Parameters: %s' % str(best_model))
    model = best_model['model']
    return model


def run(stem=False):
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train')
    x_train, vocab, idf = preprocess(df_train, stem=stem)
    df_test = read_files('./data/tweet/test')
    x_test, _, _ = preprocess(df_test, stem=stem, vocab=vocab, idf=idf)
    y_train = df_train.label.values
    y_test = df_test.label.values
    best_model = validation_train(x_train, y_train)
    best_model.summary()
    y_pred = best_model.predict(x_test)
    evaluate(y_test, y_pred.flatten() > 0.5)  # Converting probabilities to 0 and 1


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
