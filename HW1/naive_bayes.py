import os
import re
import string

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

emoticons_pattern = r'(:\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))'


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
    emoticon_tokens = re.split(emoticons_pattern, x)
    tokens = []
    for t in emoticon_tokens:
        if re.match(emoticons_pattern, t):
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
        return [tokenized_tweet.count(v) if v in tokenized_tweet else 0 for v in vocab]


def preprocess(df, stem=False, binary=True, vocab=None):
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
    if isinstance(labels, pd.Series):
        labels = labels.values
    prior = {c: np.log(labels.tolist().count(c) / len(labels)) for c in set(labels)}
    likelihood = {}
    # TODO: Add Laplace correction
    categories = list(set(labels.tolist()))
    for c in categories:
        count_wi = np.sum(features[labels == c], axis=0)
        count_all = np.sum(count_wi)
        likelihood[c] = np.log((count_wi + 1) / (count_all + len(count_wi)))
    return dict(prior=prior, likelihood=likelihood, categories=categories)


def predict(model, features):
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
    true_positives = sum(np.logical_and(y_true == true_label, y_pred == true_label))
    false_positives = sum(np.logical_and(y_true != true_label, y_pred == true_label))
    true_negatives = sum(np.logical_and(y_true != true_label, y_pred != true_label))
    false_negatives = sum(np.logical_and(y_true == true_label, y_pred != true_label))
    print('Confusion Matrix: ')
    print('\t\tTrue\tFalse')
    print('True\t%d\t\t%d' % (true_positives, false_positives))
    print('False\t%d\t\t%d' % (false_negatives, true_negatives))
    print()
    print('Accuracy = %2.2f' % (np.sum(y_true == y_pred) * 100 / len(y_pred)))
    print()


def run(stem=False, binary=True):
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
    print('Running Stemming With Frequency BoW Features')
    run(stem=True, binary=False)

    print('Running Stemming With Binary BoW Features')
    run(stem=True, binary=True)

    print('Running No Stemming With Frequency BoW Features')
    run(stem=False, binary=False)

    print('Running No Stemming With Binary BoW Features')
    run(stem=False, binary=True)


if __name__ == '__main__':
    main()
