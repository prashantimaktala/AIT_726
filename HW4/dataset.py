import os
import re

columns = ['ne', 'null', 'props', 'synt.cha', 'synt.col2', 'synt.upc', 'targets', 'words']  # 'synt.col2h'


def preprocess(lst):
    return [x.strip() for x in lst]


def boi_tags(tags):
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


def load_dataset(path='./data.wsj/train-set.txt'):
    N = 10
    sentences = []
    predicate_idx = -1
    sentence, targets = [], [[] for _ in range(N)]
    for line in open(path, 'r', encoding='utf-8'):
        if len(line.strip()) == 0:
            for t in targets:
                if len(t) != 0:
                    sentences.append((sentence, t))
            predicate_idx = -1
            sentence, targets = [], [[] for _ in range(N)]
            continue
        line = re.split(" +", line.strip())
        sentence.append((line[0], line[1]))
        if line[4] != '-':
            predicate_idx += 1
        for idx, arg in enumerate(line[5:]):
            pred = 0
            if idx == predicate_idx and line[4] != '-':
                pred = 1
            targets[idx].append((pred, arg))
    crf_format = []
    for sent in sentences:
        boi_targets = boi_tags([x[1] for x in sent[1]])
        crf_format.append([
            (token[0], token[1], target[0], target[1], boi_target) for token, target, boi_target in
            zip(sent[0], sent[1], boi_targets)
        ])
    return crf_format


if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset[0])
