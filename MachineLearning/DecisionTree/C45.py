# -*- coding:utf-8 -*
import numpy as np
import math
from collections import Counter, defaultdict


# ID3不能处理连续特征，以及ID3会倾向于选择分叉多的特征
# 因此选用信息增益比，这样对于分叉很多的特征，自己的信息熵也会很大，纯度弟，这样可以均衡特征分叉带来的偏差
# C45算法对连续值的处理是选择了一个阈值t划分成两部分


def create_data():
    X1 = np.random.rand(50, 1) * 100
    X2 = np.random.rand(50, 1) * 100
    X3 = np.random.rand(50, 1) * 100
    X4 = X1 + X2 + X3 + np.random.rand(50, 1) * 20

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

    x = np.c_[X1, X2, X3, X4, Y]
    return x, ['A', 'B', 'C', 'score']


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


def split_dataset(dataset, idx, thred=None):
    '''
    拆分函数，根据特征的取和阈值将数据集进行拆分,根据阈值是否为None来判断进行阈值划分还是特征划分
    :param dataset:
    :param idx:
    :return:
    '''
    split_data = defaultdict(list)
    if thred is None:
        for data in dataset:
            split_data[data[idx]].append(np.delete(data, idx))
        return list(split_data.values(), list(split_data.keys()))
    else:
        for data in dataset:
            split_data[data[idx] < thred].append(np.delete(data, idx))
        return list(split_data.values()), list(split_data.keys())


def info_gain(dataset, idx):
    '''
    计算信息熵
    :param dataset:
    :param idx:
    :return:
    '''
    entropy = calculate_info_entropy(dataset)
    m = len(dataset)
    split_data, _ = split_dataset(dataset, idx)
    new_entropy = 0.0
    for data in split_data:
        prob = len(data) / m
        new_entropy += prob * calculate_info_entropy(data)
    return entropy - new_entropy


def info_gain_ratio(dataset, idx, thred=None):
    '''
    计算信息熵增益比
    :param dataset:
    :param idx:
    :param thred:
    :return:
    '''
    split_data, _ = split_dataset(dataset, idx, thred)
    base_entropy = 1e-5
    m = len(dataset)
    for data in split_data:
        prob = len(data) / m
        base_entropy -= prob * math.log(prob, 2)
    return info_gain(dataset, idx) / base_entropy, thred


def get_threshold(X, idx):
    '''
    根据特征值排序后对应label是否会发生变化来选取阈值
    :param X:
    :param idx:
    :return:
    '''
    new_data = X[:, [idx, -1]].tolist()
    new_data = sorted(new_data, key=lambda x: x[0], reverse=True)
    base = new_data[0][1]
    thresholds = []
    for i in range(1, len(new_data)):
        f, l = new_data[i]
        if l != base:
            base = l
            thresholds.append(f)
    return thresholds


def choose_feature_to_split(dataset):
    n = len(dataset[0]) - 1
    m = len(dataset)

    bestGain = 0.0
    feature = -1
    thred = None
    for i in range(n):
        if not dataset[0][i].is_integer():
            threds = get_threshold(dataset, i)
            for t in threds:
                ratio, th = info_gain_ratio(dataset, i, t)
                if ratio > bestGain:
                    bestGain, feature, thred = ratio, i, t
        else:
            ratio, _ = info_gain_ratio(dataset, i)
            if ratio > bestGain:
                bestGain = ratio
                feature, thred = i, None

    return feature, thred


def create_decision_tree(dataset, feature_names):
    dataset = np.array(dataset)
    counter = Counter(dataset[:, -1])
    if len(counter) == 1:
        return dataset[0, -1]

    if len(dataset[0]) == 1:
        return counter.most_common(1)(0)(0)

    fidx, th = choose_feature_to_split(dataset)
    fname = feature_names[fidx]
    node = {fname: {'threshold': th}}
    feature_names.remove(fname)
    split_data, vals = split_dataset(dataset, fidx, th)
    for data, val in zip(split_data, vals):
        node[fname][val] = create_decision_tree(data, feature_names[:])
    return node


def classify(node, feature_names, data):
    key = list(node.keys)[0]
    node = node[key]
    idx = feature_names.index(key)

    pred = None
    thred = node['threshold']
    if thred is None:
        for key in node:
            if key != 'threshold' and data[idx] == key:
                if isinstance(node[key], dict):
                    pred = classify(node, feature_names, data)
                else:
                    pred = node[key]
    else:
        if isinstance(node[data[idx] < thred], dict):
            pred = classify(node[data[idx] < thred], feature_names, data)
        else:
            pred = node[data[idx] < thred]

    if pred is None:
        for key in node:
            if not isinstance(node[key], dict):
                pred = node[key]
                break
    return pred