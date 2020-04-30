import matplotlib.pyplot as plt
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

plt.style.use('ggplot')


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


# def sent2tokens(sent):
#     return [token for token, postag, is_predicate, _, label in sent]

def crf_cross_val_select(x_train, y_train, labels):
    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(x_train, y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    return rs


def predict_eval(rs, x_text, y_test, sorted_labels):
    crf = rs.best_estimator_
    y_pred = crf.predict(x_text)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))


def check_params(rs):
    _x = [s.parameters['c1'] for s in rs.cv_results_]
    _y = [s.parameters['c2'] for s in rs.cv_results_]
    _c = [s.mean_validation_score for s in rs.cv_results_]

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))


def main():
    train_sents = load_dataset()
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    test_sents = load_dataset('./data.wsj/test-set.txt')
    x_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    labels = list(crf.classes_)
    labels.remove('O')
    print('Labels: ' + ', '.join(labels))
    y_pred = crf.predict(x_test)
    f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    print('F1 Score: ' + str(f1_score))
    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
    rs = crf_cross_val_select(y_test, y_pred, labels)
    check_params(rs)
    predict_eval(rs, x_test, y_test, sorted_labels)


if __name__ == '__main__':
    main()
