"""
AIT726 HW 4 Due 4/30/2020

Semantic Role Labelling using CRF Model on part of conell-2005 Dataset.

Authors: Yasas, Prashanti, Ashwini

Command to train the model: python crf_model.py
Command to evaluate: perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt > evaluation_output.txt

Link to colab: https://colab.research.google.com/drive/1vZzDD_LAt35skvwKc3rYY41BmSnY5wp0#scrollTo=FuRPIfO1MPav
Link to README: https://docs.google.com/document/d/1uJ976kvGQHnLxwCvZy0k6P3TmSV5Zhzh2beAfwCLyWU/edit?usp=sharing

Flow:
i.  Create dataset
    1. Make sure all data files from assignment are in data.wsj folder.
    2. Use data.wsj/make-trainset.sh to create train-dataset (data.wsj/train-set.txt)
    3. Use data.wsj/make-testset.sh to create test-dataset (data.wsj/test-set.txt)
ii. main
    1. Load training dataset
        - dataset should be created using step i.
    2. Generate features from the data loaded
    3. Extract the target from the data loaded
    4.a. if CROSS_VALIDATE == TRUE
            4.a.1 Cross validates and prints best parameters to use for building the model.
    4.b. else
            4.b.1 Train the model using the inputs provided
            4.b.2 Make predictions using the trained model and save the outputs to `model_outputs/pred_best.txt`
            4.b.3 show training loss values for each iteration
            4.b.4 Save the model using joblib in `models/best_model.pkl`
iii. Use
        `perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt`
        command to evaluate the output file generated in main.

Features:
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

import matplotlib.pyplot as plt
import joblib

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import metrics

from preprocessing import load_dataset, conll2003_tags

plt.style.use('ggplot')

# The structure of the following code was adopted from example at `sklearn_crfsuite` official website.
# Original example was for Entity Recognition We improved over those features in `sklearn_crfsuite`
#     with target verb to indicate the target predicate.
# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

# Indicates whether to cross validate or not
CROSS_VALIDATE = False


def word2features(sent, i):
    """ Generates features for token in sentence

    :param sent: sentence
    :param i: token index
    :return: feature dictionary
    """
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
    """ Extracts features for tokens in sentence

    :param sent: input sentence
    :return: features of tokens
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    """ Extracts labels for sentence tokens

    :param sent: input sentence
    :return:  label/target values
    """
    return [s[-1] for s in sent]


def get_labels(y_train):
    """ Reads labels from train dataset

    :param y_train: training labels
    :return: list of labels
    """
    output = set()
    for t in y_train:
        output.update(t)
    output.remove('O')
    return list(output)


def crf_cross_val_select(x_train, y_train, labels):
    """ Cross-validates on the training data provided and prints the best set of hyper parameters.

    Hyper-parameters c1 and c2 determines the coefficient for L1 and L2 regularization. .

    :param x_train: Train data
    :param y_train: Target data
    :param labels: labels
    :return: RandomizedSearchCV instance
    """
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
    rs = RandomizedSearchCV(crf, params_space, cv=5, verbose=10, n_jobs=8, n_iter=5, scoring=f1_scorer)
    rs.fit(x_train, y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    return rs


def predict_save(model, sentences, path='model_outputs/pred_best.txt'):
    """ Predicts the tokens for the input sentences one by one.

    :param model: model to use for prediction.
    :param sentences: input sentences.
    :param path: path to save the predictions.
    :return: None
    """
    lines_str = ''
    for sentence, targets in sentences:
        lines = [[w[2]] for w in sentence]
        cspace = [[0] for w in sentence]
        for target in targets:
            if len(target) != 0:
                x_test = [sent2features([x[:2] + (t[0],) for x, t in zip(sentence, target)])]
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
    with open(path, 'w', encoding='utf-8') as f:
        f.write(lines_str)


def show_training_loss(model):
    """ Shows training loss values

    :param model: trained model
    :return: None
    """
    x = [t['num'] for t in model.training_log_.iterations]
    y = [t['loss'] for t in model.training_log_.iterations]
    plt.plot(x, y)
    plt.ylabel('CRF Training Loss')
    plt.xlabel('Iteration')
    plt.show()


def save_model(model, path='models/crf_best.pkl'):
    """ Saves the model.

    :param model:
    :param path:
    :return:
    """
    joblib.dump(model, path)


def load_model(path='models/crf_best.pkl'):
    """ Loads the model.

    :param path:
    :return:
    """
    return joblib.load(path)


def main():
    print("""Train-Predict-Evaluate CRF Model
    Authors: Ashwini - Prashanti - Yasas""")
    # Load dataset
    train_sents = load_dataset()
    # -- Generate features from the data loaded
    x_train = [sent2features(s) for s in train_sents]
    # -- Extract the target from the data loaded
    y_train = [sent2labels(s) for s in train_sents]
    if CROSS_VALIDATE:
        # Cross validates and prints best parameters to use for building the model.
        crf_cross_val_select(x_train, y_train, get_labels(y_train))
    else:
        # Best C1, C2 = 0.86, 0.10
        # Train the model using the inputs provided
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.86, c2=0.10,
            max_iterations=5,
            all_possible_transitions=True
        )
        model = crf.fit(x_train, y_train)
        sentences = load_dataset('./data.wsj/test-set.txt', output=None)
        # Make predictions using the trained model and save the outputs to `model_outputs/pred_best.txt`
        predict_save(crf, sentences, path='model_outputs/not_pred_best.txt')
        # show training loss values for each iteration
        show_training_loss(model)
        # Save the model using joblib in `models/best_model.pkl`
        save_model(model, path='models/not_best_model.pk')
        # $ perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt
        print('Please use `perl srl-eval.pl data.wsj/props/test.wsj.props.test model_outputs/pred_best.txt`'
              ' to evaluate the output.')


if __name__ == '__main__':
    main()
