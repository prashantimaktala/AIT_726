from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from HW4.dataset import load_dataset


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    is_predicate = sent[i][2]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'is_predicate': is_predicate,
    }
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, is_predicate, _, label in sent]


def sent2tokens(sent):
    return [token for token, postag, is_predicate, _, label in sent]


def main():
    train_sents = load_dataset()
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    test_sents = load_dataset('./data.wsj/test-set.txt')
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    labels = list(crf.classes_)
    labels.remove('O')
    print('Labels: ' + ', '.join(labels))
    y_pred = crf.predict(X_test)
    f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    print('F1 Score: ' + str(f1_score))


if __name__ == '__main__':
    main()
