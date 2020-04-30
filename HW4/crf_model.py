import matplotlib.pyplot as plt
import joblib

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import metrics

from dataset import load_dataset, conll2003_tags

plt.style.use('ggplot')

# The structure of the following code was adopted from example at `sklearn_crfsuite` official website.
# Original example was for Entity Recognition We improved over those features in `sklearn_crfsuite`
#     with target verb to indicate the target predicate.
# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

""" Features
The function `word2features` produces the features for a given (sentence, predicate) pair.
input:  (sent: List[Tuple[token: str, pos_tag: str, is_predicate: boolean]], i)
    sent - input sentence for a specific predicate
    i - index of current word to create the features for
output: features: Dict
    bias - Word independent feature to capture the label bias in the dataset when training. Always set to 1.
    word.lower() - Lowercased word.
    word[-3:] - Last three charactors of word
    word[-2:] - Last two charactors of word
    word.isupper() - Whether all characters the word are in capital letters.
    word.istitle() - Whether the word is in title cased.
    word.isdigit() - Whether the word is a digit.
    postag - POS tag.
    postag[:2] - First two characters of POS tag.
    is_predicate - Whether this word is the predicate.
    -1:word.lower() - Previous lowercased word.
    -1:word.istitle() - Whether the previous word is in title cased.
    -1:word.isupper() - Whether all characters in the previous word are in capital letters.
    -1:postag - POS tag of previous word.
    -1:postag[:2] - First two characters of previous POS tag.
    +1:word.lower() - Next lowercased word.
    +1:word.istitle() - Whether the next word is in title cased.
    +1:word.isupper() - Whether all characters in the next word are in capital letters.
    +1:postag - POS tag of next word.
    +1:postag[:2] - First two characters of next POS tag.
    BOS - Beginning of Sentence Indicator.
    EOS - End of Sentence Indicator.
"""


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
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [s[-1] for s in sent]


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
                            cv=5,
                            verbose=10,
                            n_jobs=8,
                            n_iter=5,
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


def predict_save(model, sentences):
    lines_str = ''
    for sentense, targets in sentences:
        lines = [[w[2]] for w in sentense]
        cspace = [[0] for w in sentense]
        for target in targets:
            if len(target) != 0:
                x_test = [sent2features([x[:2] + (t[0],) for x, t in zip(sentense, target)])]
                y_pred = model.predict(x_test)
                y_pred = list(conll2003_tags(y_pred))[0]
                for idx, (y_pred, clen) in enumerate(y_pred):
                    lines[idx] += [y_pred]
                    cspace[idx] += [clen]
        for i, line in enumerate(lines):
            lines_str += line[0]
            pre_len = 0
            for j, p in enumerate(line[1:]):
                lines_str += ' ' * (((24 - len(line[0])) if j == 0 else (16 - pre_len)) - cspace[i][j + 1])
                lines_str += p
                pre_len = len(p) - cspace[i][j + 1]
            lines_str += '\n'
        lines_str += '\n'
    with open('model_outputs/pred.txt', 'w', encoding='utf-8') as f:
        f.write(lines_str)


def save_model(model, path='models/best_model.pkl'):
    joblib.dump(model, path)


def load_model(path='models/best_model.pkl'):
    return joblib.load(path)


def main():
    print(''' Train-Predict-Evaluate CRF Model
    Authors: Ashwini - Prashanti - Yasas
    ''')
    train_sents = load_dataset()
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.86, c2=0.10,
        max_iterations=100,
        all_possible_transitions=True
    )
    model = crf.fit(x_train, y_train)
    sentences = load_dataset('./data.wsj/test-set.txt', output=None)
    predict_save(crf, sentences)
    save_model(model)
    # $ perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt
    print('Please use `perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt`'
          ' to evaluate the output.')


if __name__ == '__main__':
    main()
