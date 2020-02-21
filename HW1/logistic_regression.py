import os
import re
import string

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import logging

logging.basicConfig(filename='logistic_regression_results.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

stemmer = PorterStemmer()

# regex from https://stackoverflow.com/questions/28077049/regex-matching-emoticons
emoticons_re = r'(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)'


def read_files(path):
    corpus = []
    for label in ['negative', 'positive']:
        tmp = os.path.join(path, label)
        for file in os.listdir(tmp):
            with open(os.path.join(tmp, file), 'r', encoding='utf-8') as f:
                tweet = f.read().strip()
                corpus.append((tweet, 1 if label == 'positive' else 0))
    df = pd.DataFrame.from_records(corpus, columns=['tweet', 'label'])
    return df


def tokenize(x, stem=False):
    x = re.sub(r'(?:<[^>]+>)', '', x)
    x = re.sub('([A-Z][a-z]+)', lambda t: t.group(0).lower(), x)
    emoticon_tokens = re.split(emoticons_re, x)
    tokens = []
    for t in emoticon_tokens:
        if re.match(emoticons_re, t):
            tokens.append(t)
        else:
            tokens += word_tokenize(x)
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def vocabulary(tokenized_tweets):
    vocab = set()
    for tokens in tokenized_tweets:
        vocab.update(tokens)
    return list(vocab)


def bag_of_words(tokenized_tweet, vocab, binary=True):
    if binary:
        return [1 if v in tokenized_tweet else 0 for v in vocab]
    else:
        return [tokenized_tweet.count(v) for v in vocab]


def tf_idf(collection, vocab, idf=None):
    tf = [[doc.count(v) for v in vocab] for doc in collection]
    tf = np.array([[1 + np.log(count) if count > 0 else 0 for count in doc] for doc in tf])
    if idf is None:
        N = len(collection)
        df = [len([1 for doc in collection if w in doc]) for w in vocab]
        idf = np.array([np.log(N / t) for t in df])
    output = tf * idf
    return output, idf


def preprocess(df, stem=False, use_tfidf=False, binary=True, vocab=None, idf=None):
    if stem:
        tokens = df.tweet.apply(lambda x: tokenize(x, True))
    else:
        tokens = df.tweet.apply(tokenize)
    if vocab is None:
        vocab = vocabulary(tokens)
    if use_tfidf:
        output, idf = tf_idf(tokens, vocab, idf)
    else:
        if binary:
            output = tokens.apply(lambda x: bag_of_words(x, vocab))
        else:
            output = tokens.apply(lambda x: bag_of_words(x, vocab, False))
    if not isinstance(output, np.ndarray):
        output = np.array(output.values.tolist())
    if use_tfidf:
        return output, vocab, idf
    return output, vocab


# TODO: Check
def sigmoid(x):
    """ Sigmoid Function

    :param x: var
    :return: Sigmoid value
    """
    return 1 / (1 + np.exp(-x))


# TODO: Check
def pred_proba(model, features):
    """ Predict the (softmax) probability of each document being positive.

    :param model: a trained model
    :param features: feature matrix of shape D x V.
    :return: Predicted +ve probability of documents.
    """
    w, b = model['w'], model['b']
    return np.array([sigmoid(np.dot(w, x) + b) for x in features])


# TODO: Implement
def predict(model, features, threshold=0.5):
    return pred_proba(model, features) > threshold


# TODO: Implement
def cross_entropy_loss(y, y_hat):
    return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# TODO: Implement
def cost(model, features, labels):
    y_pred = pred_proba(model, features)
    m = labels.shape[0]
    result = np.sum([cross_entropy_loss(y, y_hat) for y, y_hat in zip(labels, y_pred)]) / m
    return result


# TODO: Implement
def train(features, labels, n_iter=20, batch_size=10, learning_rate=0.05, penalty=None, alpha=0.001):
    num_features = features.shape[1]
    model = {'w': np.zeros(num_features), 'b': 0.0}
    m = labels.shape[0]
    cost_lst = []
    for epoch in range(n_iter):
        idxes = np.random.permutation(m)
        x_shuffled, y_shuffled = features[idxes], labels[idxes]
        # mini-batch gradient descent
        for batch in range(0, m, batch_size):
            x_batch, y_batch = x_shuffled[batch:batch + batch_size], y_shuffled[batch:batch + batch_size]
            m_batch = y_batch.shape[0]
            z = pred_proba(model, x_batch)  # estimated y
            r_w = 0
            #  regularization
            if isinstance(penalty, str):
                if penalty.lower() == 'l2':  # Ridge Regression
                    # L2 - euclidean distance from the origin
                    r_w = model['w']
            # stochastic gradient ascent
            model['w'] += learning_rate * (np.dot(np.transpose(x_batch), (y_batch - z).reshape(-1)).reshape(
                -1) / m_batch - alpha * r_w / m_batch)
            model['b'] += learning_rate * np.sum(y_batch - z) / m_batch
        cost_val = cost(model, features, labels)
        cost_lst.append(cost_val)
        logging.info('epoch = {} & cost = {}'.format(epoch + 1, cost_val))
    return model


def evaluate(y_true, y_pred, true_label=1):
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


# noinspection DuplicatedCode
def run(stem=False, mode='binary'):
    df_train = read_files('./data/tweet/train')
    idf = None
    if mode == 'binary':
        x_train, vocab = preprocess(df_train, stem=stem, binary=True)
    elif mode == 'tfidf':
        x_train, vocab, idf = preprocess(df_train, stem=stem, use_tfidf=True)
    else:
        x_train, vocab = preprocess(df_train, stem=stem, binary=False)
    y_train = df_train.label.values
    model = train(x_train, y_train, penalty='l2')
    df_test = read_files('./data/tweet/test')
    if mode == 'binary':
        x_test, _ = preprocess(df_test, stem=stem, binary=True, vocab=vocab)
    elif mode == 'tfidf':
        x_test, _, _ = preprocess(df_test, stem=stem, use_tfidf=True, vocab=vocab, idf=idf)
    else:
        x_test, _ = preprocess(df_test, stem=stem, binary=False, vocab=vocab)
    y_pred = predict(model, x_test)
    y_test = df_test.label.values
    evaluate(y_test, y_pred)


def validate(stem=False, mode='binary', n_iter=20, batch_size=10, learning_rate=0.05, penalty=None, alpha=0.001):
    df = read_files('./data/tweet/train')
    df_train, df_test = df[:round(df.shape[0] * 0.8)], df[round(df.shape[0] * 0.8):]
    idf = None
    if mode == 'binary':
        x_train, vocab = preprocess(df_train, stem=stem, binary=True)
    elif mode == 'tfidf':
        x_train, vocab, idf = preprocess(df_train, stem=stem, use_tfidf=True)
    else:
        x_train, vocab = preprocess(df_train, stem=stem, binary=False)
    y_train = df_train.label.values
    model = train(x_train, y_train, n_iter=n_iter, batch_size=batch_size, learning_rate=learning_rate, penalty=penalty,
                  alpha=alpha)
    if mode == 'binary':
        x_test, _ = preprocess(df_test, stem=stem, binary=True, vocab=vocab)
    elif mode == 'tfidf':
        x_test, _, _ = preprocess(df_test, stem=stem, use_tfidf=True, vocab=vocab, idf=idf)
    else:
        x_test, _ = preprocess(df_test, stem=stem, binary=False, vocab=vocab)
    y_pred = predict(model, x_test)
    y_test = df_test.label.values
    evaluate(y_test, y_pred)


def main():
    logging.info('AIT_726 Logistic Regression Output')
    logging.info('Authors: Yasas, Prashanti , Ashwini')
    for penalty in [None, 'l2']:
        if penalty is None:
            logging.info('Evaluating without Regularization')
        else:
            logging.info('Evaluating with L2 Regularization')
        logging.info('Running Stemming With Frequency BoW Features')
        run(stem=True, mode='freq')
        logging.info('Running Stemming With Binary BoW Features')
        run(stem=True, mode='binary')
        logging.info('Running No Stemming With Frequency BoW Features')
        run(stem=False, mode='freq')
        logging.info('Running Stemming With TFIDF Features')
        run(stem=True, mode='tfidf')
        logging.info('Running No Stemming With Binary BoW Features')
        run(stem=False, mode='binary')
        logging.info('Running No Stemming With TFIDF Features')
        run(stem=False, mode='tfidf')
        logging.info('')


def main_validate():
    validate(stem=False, mode='binary', n_iter=20, batch_size=10, learning_rate=0.05, penalty=None, alpha=0.001)


if __name__ == '__main__':
    main()
