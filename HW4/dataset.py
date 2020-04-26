import os
import re

columns = ['ne', 'null', 'props', 'synt.cha', 'synt.col2', 'synt.upc', 'targets', 'words']  # 'synt.col2h'


def preprocess(lst):
    return [x.strip() for x in lst]


def load_dataset(path='./data.wsj/test-set.txt'):
    NUM_TARGETS = 10
    ds = []
    sentence, targets = [], [[] for _ in range(NUM_TARGETS)]
    for line in open(path, 'r', encoding='utf-8'):
        if len(line.strip()) == 0:
            ds.append((sentence, targets))
            sentence, targets = [], [[] for _ in range(NUM_TARGETS)]
            continue
        line = re.split(" +", line.strip())
        sentence.append(line[0])
        for idx, arg in enumerate(line[5:]):
            try:
                targets[idx].append((line[4], arg))
            except:
                print(idx)
    return ds


if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset[1])
