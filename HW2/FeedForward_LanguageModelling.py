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
from nltk.util import ngrams

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
    df = pd.DataFrame(corpus)  # build dataframe
    return df

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
            tokens.append(t)  # append emoticons without work tokenize
        else:
            tokens += word_tokenize(x)
    return tokens

def preprocess(df):
    """
    preprocess - calls appropriate tokenize to generate training and test data with required vocabulary
    """
    tokens = df.tweet.apply(tokenize)
    bigrams = ngrams(tokens, 2)
    return np.array(list(bigrams))


def run():
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train/positive')
    x_train_pos = preprocess(df_train)
    df_test = read_files('./data/tweet/test/positive')
    x_test_pos = preprocess(df_test)

    # y_train = df_train.label.values
    # model = train(x_train, y_train)



def main():
    """
    main - runs all the modules via run function
    """
    logging.info('AIT_726 Feed Forward Neural Network')
    logging.info('Authors: Yasas, Prashanti , Ashwini')
    logging.info(' TFIDF with stemming')
    run()

    logging.info('TFIDF without stemming')
    # run(stem=False)


if __name__ == '__main__':
    main()