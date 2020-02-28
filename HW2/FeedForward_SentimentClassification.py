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
from nltk.stem.porter import PorterStemmer


# use logging to save the results
logging.basicConfig(filename='naive_bayes_results.log', level=logging.INFO)
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


def run(stem=False, binary=True):
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train')
    x_train, vocab = preprocess(df_train, stem=stem, binary=binary)
    y_train = df_train.label.values

def main():
    """
    main - runs all the modules via run function
    """
    # logging.info('AIT_726 Feed Forward Neural Network')
    # logging.info('Authors: Yasas, Prashanti , Ashwini')
    # logging.info('Running Stemming With Frequency BoW Features')
    run(stem=True, binary=False)

    # logging.info('Running Stemming With Binary BoW Features')
    # run(stem=True, binary=True)
    #
    # logging.info('Running No Stemming With Frequency BoW Features')
    # run(stem=False, binary=False)
    #
    # logging.info('Running No Stemming With Binary BoW Features')
    # run(stem=False, binary=True)


if __name__ == '__main__':
    main()