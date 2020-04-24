import logging
from collections import Counter

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from node import _Node
from decision_tree import DecisionTree

class RandomForest(ClassifierMixin):
    def __init__(self, n_trees=10):
        self._n_trees = n_trees

        self._forest = [None] * self._n_trees
        self._using_features = [None] * self._n_trees

    def fit(self, X, y):
        self._classes = np.unique(y)
        # 各決定木にわたすデータは、元データをブートストラップサンプル (復元ありの抽出) および特徴量をランダムに選択
        bootstrapped_X, bootstrapped_y = self._bootstrap_sample(X, y)
        for i, (i_bootstrapped_X, i_bootstrapped_y) in enumerate(zip(bootstrapped_X, bootstrapped_y)):
            tree = DecisionTree()
            tree.fit(i_bootstrapped_X, i_bootstrapped_y)
            self._forest[i] = tree

    def _bootstrap_sample(self, X, y):
        #与えられたデータをブートストラップサンプル (復元抽出)
        # 同時に、特徴量方向のサンプリング
        n_features = X.shape[1]
        n_features_forest = np.floor(np.sqrt(n_features))
        bootstrapped_X = list()
        bootstrapped_y = list()
        for i in range(self._n_trees):
            ind = np.random.choice(len(y), size=len(y))  # 用いるサンプルをランダムに選択
            col = np.random.choice(n_features, size=int(n_features_forest), replace=False)  # 用いる特徴量をランダムに選択
            bootstrapped_X.append(X[np.ix_(ind, col)])
            bootstrapped_y.append(y[ind])
            self._using_features[i] = col
        return bootstrapped_X, bootstrapped_y

    def predict(self, X):
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self._forest[0] is None:
            raise ValueError('fitしてね')
        votes = [tree.predict(X[:, using_feature]) for tree, using_feature in zip(self._forest, self._using_features)]  # n_trees x n_samples
        counts = [Counter(row) for row in np.array(votes).transpose()]  # n_samples だけの Counter オブジェクト
        # 各 tree の意見の集計
        counts_array = np.zeros((len(X), len(self._classes)))  # n_samples x n_classes
        for row_index, count in enumerate(counts):
            for class_index, class_ in enumerate(self._classes):
                counts_array[row_index, class_index] = count[class_]
        proba = counts_array / self._n_trees  # 規格化する
        return proba

if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dt = DecisionTree()
    dt.fit(X_train, y_train)

    rf = RandomForest()
    rf.fit(X_train, y_train)

    print('DecisionTree: ')

    # dt_predicted_y_train = dt.predict(X_train)
    # print('  predicted_y_train: {}'.format(dt_predicted_y_train))
    # print('  (actual)         : {}'.format(y_train))
    print('  score_train: {}'.format(dt.score(X_train, y_train)))
    # dt_predicted_y_test = dt.predict(X_test)
    # print('  predicted_y_test: {}'.format(dt_predicted_y_test))
    # print('  (actual)        : {}'.format(y_test))
    print('  score_test: {}'.format(dt.score(X_test, y_test)))

    print('RandomForest: ')

    # rf_predicted_y_train = rf.predict(X_train)
    # print('  predicted_y_train: {}'.format(rf_predicted_y_train))
    # print('  (actual)         : {}'.format(y_train))
    print('  score_train: {}'.format(rf.score(X_train, y_train)))
    # rf_predicted_y_test = rf.predict(X_test)
    # print('  predicted_y_test: {}'.format(rf_predicted_y_test))
    # print('  (actual)        : {}'.format(y_test))
    print('  score_test: {}'.format(rf.score(X_test, y_test)))

    print('Scikit-learn RandomForest: ')

    ret = RandomForestClassifier().fit(X_train, y_train)
    print('  score_train: {}'.format(ret.score(X_train, y_train)))
    print('  score_test: {}'.format(ret.score(X_test, y_test)))
