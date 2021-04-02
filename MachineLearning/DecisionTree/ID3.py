import numpy as np
import math
from collections import Counter, defaultdict


def create_data():
    X1 = np.random.rand(50, 1) * 100
    X2 = np.random.rand(50, 1) * 100
    X3 = np.random.rand(50, 1) * 100

    def f(x):
        return 2 if x > 70 else 1 if x > 40 else 0

    y = X1 + X2 + X3
    Y = y > 150
    Y = Y + 0

    r = map(f, X1)
    X1 = list(r)

    r = map(f, X2)
    X2 = list(r)

    r = map(f, X3)
    X3 = list(r)

    x = np.c_[X1, X2, X3, Y]
    return x, ['A', 'B', 'C']



def calculate_info_entropy(dataset):
    '''
    计算信息熵
    :param dataset:
    :return:
    '''
    n = len(dataset)
    lables = Counter(dataset[:, -1])
    entropy = 0.0
    for k, v in lables.items():
        prob = v / n
        entropy -= prob * math.log(prob, 2)

    return entropy


def split_dataset(dataset, idx):
    '''
    拆分函数，根据特征的取值将数据集进行拆分
    :param dataset:
    :param idx:
    :return:
    '''
    split_data = defaultdict(list)
    for data in dataset:
        split_data[data[idx]].append(np.delete(data, idx))
    return list(split_data.values())


def choose_feature_to_split(dataset):
    '''
    选择信息增益最大的特征
    :param dataset:
    :return:
    '''
    n = len(dataset[0]) - 1
    m = len(dataset)

    entropy = calculate_info_entropy(dataset)
    bestGain = 0.0
    feature = -1

    for i in range(n):
        split_data = split_dataset(dataset, i)
        new_entropy = 0.0
        for data in split_data:
            prob = len(data) / m
            new_entropy += prob * calculate_info_entropy(data)
        gain = entropy - new_entropy
        if gain > bestGain:
            bestGain = gain
            feature = i
    return feature


def create_decision_tree(dataset, feature_names):
    dataset = np.array(dataset)
    counter = Counter(dataset[:, -1])

    if len(counter) == 1:
        return dataset[0, -1]


    if len(dataset[0]) == 1:
        return counter.most_common(1)[0][0]

    fidx = choose_feature_to_split(dataset)
    fname = feature_names[fidx]
    node = {fname: {}}
    feature_names.remove(fname)

    # 递归调用，对每一个切分出来的取值递归建树
    split_data, vals = split_dataset(dataset, fidx)
    for data, val in zip(split_data, vals):
        node[fname][val] = create_decision_tree(np.array(data), feature_names[:])
    return node


def classify(node, feature_names, data):
    key = list(node.keys)[0]
    node = node[key]
    idx = feature_names.index(key)

    pred = None
    for key in node:
        if data[idx] == key:
            if isinstance(node[key], dict):
                pred = classify(node[key], feature_names, data)
            else:
                pred = node[key]

    if pred == None:
        for key in node:
            if not isinstance(node[key], dict):
                pred = node[key]
                break

    return pred