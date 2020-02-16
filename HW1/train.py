import os
import re
import string

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


def preprocess(df):
    df['tokens'] = df.tweet.apply(tokenize)
    df['stemmed_tokens'] = df.tweet.apply(lambda x: tokenize(x, True))
    vocab = vocabulary(df.tokens)
    stemmed_vocab = vocabulary(df.stemmed_tokens)
    # Binary
    bow_bin = df.tokens.apply(lambda x: bag_of_words(x, vocab))
    bow_stemmed_bin = df.tokens.apply(lambda x: bag_of_words(x, vocab))
    # Frequency
    bow_freq = df.tokens.apply(lambda x: bag_of_words(x, vocab, False))
    bow_stemmed_freq = df.tokens.apply(lambda x: bag_of_words(x, vocab, False))
    print(vocab)
    return df


if __name__ == '__main__':
    df_train = read_files('./data/tweet/train')
    df_train = preprocess(df_train)
    print(df_train.columns)
