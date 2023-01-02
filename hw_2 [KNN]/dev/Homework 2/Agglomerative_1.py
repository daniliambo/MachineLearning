from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
# import cv2
from collections import deque
from typing import NoReturn


# Task 3

class AgglomerativeClustering:

    def __init__(self, n_clusters: int = 2, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        distances = np.sum(X ** 2, axis=1).reshape(-1, 1) - 2 * np.matmul(X, X.T) + np.sum(X ** 2, axis=1)
        np.fill_diagonal(distances, np.nan)
        classes = np.arange(X.shape[0], dtype=np.int64)
        for _ in range(self.n_clusters, X.shape[0]):
            pairs = np.unravel_index(np.nanargmin(distances), distances.shape)
            classes[classes == pairs[1]] = pairs[0]
            if self.linkage == 'average':
                _data1, _data2 = np.nan_to_num(distances[pairs[0], :]), np.nan_to_num(distances[pairs[1], :])
                _power1, _power2 = np.count_nonzero(_data1), np.count_nonzero(_data2)
                _data = (_power1 * _data1 + _power2 * _data2) / (_power1 + _power2)
                _data[_data == 0] = np.nan
                distances[:, pairs[0]] = distances[pairs[0], :] = _data
                distances[pairs[0]][pairs[0]] = np.nan
                distances[pairs[1], :] = distances[:, pairs[1]] = np.nan
            elif self.linkage == 'single':
                _data = np.where(distances[pairs[0], :] < distances[pairs[1], :],
                                 distances[pairs[0], :], distances[pairs[1], :])
                distances[:, pairs[0]] = distances[pairs[0], :] = _data
                distances[pairs[0]][pairs[0]] = np.nan
                distances[pairs[1], :] = distances[:, pairs[1]] = np.nan
            elif self.linkage == 'complete':
                _data = np.where(distances[pairs[0], :] > distances[pairs[1], :],
                                 distances[pairs[0], :], distances[pairs[1], :])
                distances[:, pairs[0]] = distances[pairs[0], :] = _data
                distances[pairs[0]][pairs[0]] = np.nan
                distances[pairs[1], :] = distances[:, pairs[1]] = np.nan
        mapping = np.unique(classes)
        mapping = dict(zip(mapping, range(len(mapping))))
        return np.vectorize(mapping.get)(classes)


np.random.seed(42)


def visualize_clusters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[l] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


# tests
X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_2, true_labels_2 = make_moons(400, noise=0.075)

X = X_1[:4]
ac = AgglomerativeClustering()
res = ac.fit_predict(X)
print(res)
