"""
AIT726 HW 1 Due 2/20/2020
Sentiment classification using Naive Bayes and Logistic Regression on a dataset of 4181 training and 4182 testing tweets.
Authors: Yasas, Prashanti, Ashwini
Command to run the file: naive_bayes.py

Flow:
i. main
ii. run  default parameters: stem = false, binary = true
    1.Train the model
        a. Read the dataset
        b. Perform preprocessing
            - Build vocab
        c. Train the model
            - calculate prior and likelihood
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

stemmer = PorterStemmer()

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
            with open(os.path.join(tmp, file), 'r', encoding='utf-8') as f:   #join path to read all the tweets
                tweet = f.read().strip()     # strip function to remove leading and trailing spaces in each tweet
                corpus.append((tweet, 1 if label == 'positive' else 0))
    df = pd.DataFrame.from_records(corpus, columns=['tweet', 'label'])  #build dataframe
    return df


def tokenize(x, stem=False):
    """
    tokenize function takes care of handling removal of html tags, conversion of capitalized words to lowercase except
    for all capital words, handling of emoticons. we have created streams of tokens without stemming using word_tokenize
    as well as tokens with stemming using PotterStemmer.
    """
    x = re.sub(r'(?:<[^>]+>)', '', x)  #substitute html tags
    x = re.sub('([A-Z][a-z]+)', lambda t: t.group(0).lower(), x)  # group 0 refers to A-Z, lowercase group 0
    emoticon_tokens = re.split(emoticons_re, x)  # separate emoticons
    tokens = []
    for t in emoticon_tokens:
        if re.match(emoticons_re, t):
            tokens.append(t)   # append emoticons without work tokenize
        else:
            tokens += word_tokenize(x)
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]  #perform stemming
    return tokens


def vocabulary(tokenized_tweets):
    """
    vocabulary - we have created word by word vocabulary for the complete training data for all the provided tokens
    """
    vocab = set()
    for tokens in tokenized_tweets:
        vocab.update(tokens)
    return list(vocab)   #create complete list of tokens for all the tweets


def bag_of_words(tokenized_tweet, vocab, binary=True):
    """
    bag_of_words - we have created both binary and frequency count representation of BOW
    """
    if binary:
        return [1 if v in tokenized_tweet else 0 for v in vocab]
    else:
        return [tokenized_tweet.count(v) if v in tokenized_tweet else 0 for v in vocab]


def preprocess(df, stem=False, binary=True, vocab=None):
    """
    preprocess - calls appropriate tokenize to generate training and test data with required vocabulary
    """
    if stem:
        tokens = df.tweet.apply(lambda x: tokenize(x, True))
    else:
        tokens = df.tweet.apply(tokenize)
    if vocab is None:
        vocab = vocabulary(tokens)
    if binary:
        output = tokens.apply(lambda x: bag_of_words(x, vocab))
    else:
        output = tokens.apply(lambda x: bag_of_words(x, vocab, False))
    return np.array(output.values.tolist()), vocab


def train(features, labels):
    """
    train - calculates the class prior and likelihoods to be used in Naive Bayes predictions
    """
    if isinstance(labels, pd.Series):
        labels = labels.values
    prior = {c: np.log(labels.tolist().count(c) / len(labels)) for c in set(labels)}
    likelihood = {}
    # Add Laplace correction
    categories = list(set(labels.tolist()))
    for c in categories:
        count_wi = np.sum(features[labels == c], axis=0)
        count_all = np.sum(count_wi)
        likelihood[c] = np.log((count_wi + 1) / (count_all + len(count_wi)))
    return dict(prior=prior, likelihood=likelihood, categories=categories)


def predict(model, features):
    """
     predict - predicts the class of all of the test documents for all of the feature vectors using Naive Bayes
     """
    y_log_prob = None
    for c in model['categories']:
        y_log_prob_c = model['prior'][c] + np.sum(features * model['likelihood'][c], axis=1)
        if y_log_prob is None:
            y_log_prob = y_log_prob_c.reshape((-1, 1))
        else:
            y_log_prob = np.append(y_log_prob, y_log_prob_c.reshape((-1, 1)), 1)
    y_argmax = np.argmax(y_log_prob, axis=1)
    y_pred = np.array([model['categories'][x] for x in y_argmax])
    return y_pred


def evaluate(y_true, y_pred, true_label=1):
    """
     evaluate - calculates and prints accuracy and confusion matrix for predictions
     """
    true_positives = sum(np.logical_and(y_true == true_label, y_pred == true_label))
    false_positives = sum(np.logical_and(y_true != true_label, y_pred == true_label))
    true_negatives = sum(np.logical_and(y_true != true_label, y_pred != true_label))
    false_negatives = sum(np.logical_and(y_true == true_label, y_pred != true_label))
    logging.info('Confusion Matrix: ')
    logging.info('\t\tTrue\tFalse')
    logging.info('True\t%d\t\t%d' % (true_positives, false_positives))
    logging.info('False\t%d\t\t%d' % (false_negatives, true_negatives))
    logging.info('Accuracy = %2.2f' % (np.sum(y_true == y_pred) * 100 / len(y_pred)))
    logging.info('')


def run(stem=False, binary=True):
    """
    run - Execution of appropriate functions as per the required call
    """
    df_train = read_files('./data/tweet/train')
    x_train, vocab = preprocess(df_train, stem=stem, binary=binary)
    y_train = df_train.label.values
    model = train(x_train, y_train)
    df_test = read_files('./data/tweet/test')
    x_test, _ = preprocess(df_test, stem=stem, binary=binary, vocab=vocab)
    y_pred = predict(model, x_test)
    y_test = df_test.label.values
    evaluate(y_test, y_pred)


def main():
    """
    main - runs all the modules via run function
    """
    logging.info('AIT_726 Naive_bayes Output')
    logging.info('Authors: Yasas, Prashanti , Ashwini')
    logging.info('Running Stemming With Frequency BoW Features')
    run(stem=True, binary=False)

    logging.info('Running Stemming With Binary BoW Features')
    run(stem=True, binary=True)

    logging.info('Running No Stemming With Frequency BoW Features')
    run(stem=False, binary=False)

    logging.info('Running No Stemming With Binary BoW Features')
    run(stem=False, binary=True)


if __name__ == '__main__':
    main()
