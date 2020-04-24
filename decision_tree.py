import logging
from collections import Counter

import numpy as np
from sklearn.base import ClassifierMixin

from node import _Node

# ClassifierMixinを継承することでsklearn由来のメソッド(fit, predict)を利用可能

# 2分決定木
class DecisionTree(ClassifierMixin):
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self._classes = np.unique(y)
        self._tree = self._grow(X, y)

    def _grow(self, X, y):
        # 深さ優先で再帰的に木を成長させる
        uniques, counts = np.unique(y, return_counts=True)  # 各クラスの出現回数を数える
        counter = dict(zip(uniques, counts))
        class_count = [counter[c] if c in counter else 0 for c in self._classes]
        this = _Node(class_count) # 自分自身を生成

        # サンプルが一つならLeaf
        if len(y) == 1:
            return this
        
        # 全て同じクラスならLeaf
        # all() : 引数に指定したイテラブルオブジェクトの要素がすべてTrueならTrue
        if all(y[0] == y):
            return this

        # サンプルが全部同じ特徴量を保つ場合は分岐不可能なので葉ノードを返して終了
        if (X[0] == X).all():
            return this

        # 以降は分岐ノードで有ることが確定
        left_X, left_y, right_X, right_y, feature_id, threshold = self._branch(X, y)
        this.feature_id = feature_id
        this.threshold = threshold
        this.left = self._grow(left_X, left_y) # 左側の木を成長
        this.right = self._grow(right_X, right_y) # 右側の木を成長

        return this

    def _branch(self, X, y):
        # ジニ係数に従ってサンプル分割
        gains = list()
        rules = list()

        for feature_id, xs, in enumerate(X.transpose()):
            thresholds = self._get_branching_threshold(xs)
            for th in thresholds:
                left_y = y[xs < th]
                right_y = y[th <= xs]
                gain = self._delta_gini_index(left_y, right_y) # この分割によるジニ係数の減少量(小さい方が偉い)
                gains.append(gain)
                rules.append((feature_id, th))

        best_rule = rules[np.argmin(gains)] # ジニ係数的に最も正しいルール
        feature_id = best_rule[0]
        threshold = best_rule[1]
        split = X[:, feature_id] < threshold # 閾値による分割を取得
        return X[split], y[split], X[~split], y[~split], feature_id, threshold

    def _get_branching_threshold(self, xs):
        # xs の分岐条件となる閾値を全取得
        unique_xs = np.unique(xs) # np.unique()はソート済みの結果を返すこと注意
        return (unique_xs[1:] + unique_xs[:-1]) / 2 # [3, 4, 6] => [3.5, 5.0]

    def _delta_gini_index(self, left, right):
        # ジニ係数の減少量を計算する(小さい方が偉い)
        n_left = len(left)
        n_right = len(right)
        n_total = n_left + n_right

        # 左側
        _, counts = np.unique(left, return_counts=True)  # 各クラスの出現回数を数えて
        left_ratio_classes = counts / n_left  # 割合にする
        left_gain = (n_left / n_total) * (1 - (left_ratio_classes ** 2).sum())
        # 右側
        _, counts = np.unique(right, return_counts=True)  # 各クラスの出現回数を数えて
        right_ratio_classes = counts / n_right  # 割合にする
        right_gain = (n_right / n_total) * (1 - (right_ratio_classes ** 2).sum())

        return left_gain + right_gain


    def predict(self, X):
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self._tree is None:
            raise ValueError('fitしてね')
        return np.array([self._predict_one(xs) for xs in X])

    def _predict_one(self, xs):
        """1サンプルを予測"""
        node = self._tree
        while not node.is_leaf:  # 葉ノードに到達するまで繰り返す
            is_left = xs[node.feature_id] < node.threshold  # True: left, False: right
            node = node.left if is_left else node.right
        class_count = node.class_count
        return np.array(class_count) / sum(class_count)
