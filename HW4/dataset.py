import os
import re

columns = ['ne', 'null', 'props', 'synt.cha', 'synt.col2', 'synt.upc', 'targets', 'words']  # 'synt.col2h'


def preprocess(lst):
    """ Preprocess text input (lines)

    :param lst:
    :return:
    """
    return [x.strip() for x in lst]


def boi_tags(tags):
    """ Converts

    :param tags:
    :return:
    """
    result = []
    last_tag = 'O'
    for t in tags:
        if t.startswith('(') and t.endswith(')'):
            tag = 'B_' + t[1:-1].strip('*')
            last_tag = 'O'
        elif t.startswith('('):
            tag = t[1:-1].strip('*')
            last_tag = 'I_' + tag
            tag = 'B_' + tag
        elif t.endswith(')'):
            tag = last_tag
            last_tag = 'O'
        else:
            tag = last_tag
        result.append(tag)
    return result


def conll2003_tags(y_pred):
    """

    :param y_pred:
    :return:
    """
    for s_pred in y_pred:
        cdatas = []
        for t_pred, t_pred_next in zip(s_pred, s_pred[1:] + [None]):
            clen = 0
            cdata = ''
            if t_pred.startswith('I_'):
                cdata = '*'
            elif t_pred.startswith('B_'):
                cdata = '(' + t_pred[2:] + '*'
            else:
                cdata = '*'
            clen = len(cdata) - 1
            if t_pred != 'O':
                if t_pred_next is None or not t_pred_next.startswith('I_'):
                    cdata += ')'
            cdatas += [(cdata, clen)]
        yield cdatas


def load_dataset(path='./data.wsj/train-set.txt', output='crf'):
    N = 10
    sentences = []
    predicate_idx = -1
    sentence, targets = [], [[] for _ in range(N)]
    for line in open(path, 'r', encoding='utf-8'):
        if len(line.strip()) == 0:
            if output == 'crf':
                for t in targets:
                    if len(t) != 0:
                        sentences.append((sentence, t))
            else:
                sentences.append((sentence, targets))
            predicate_idx = -1
            sentence, targets = [], [[] for _ in range(N)]
            continue
        line = re.split(" +", line.strip())
        if output == 'crf':
            sentence.append((line[0], line[1]))
        else:
            sentence.append((line[0], line[1], line[4]))
        if line[4] != '-':
            predicate_idx += 1
        for idx, arg in enumerate(line[5:]):
            pred = 0
            if idx == predicate_idx and line[4] != '-':
                pred = 1
            targets[idx].append((pred, arg))
    if output == 'crf':
        crf_format = []
        for sent in sentences:
            boi_targets = boi_tags([x[1] for x in sent[1]])
            crf_format.append([
                (token[0], token[1], target[0], target[1], boi_target) for token, target, boi_target in
                zip(sent[0], sent[1], boi_targets)
            ])
        return crf_format
    return sentences


if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset[0])
