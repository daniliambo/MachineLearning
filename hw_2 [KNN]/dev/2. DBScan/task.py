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


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps, self.min_samples, self.leaf_size, self.metric = eps, min_samples, leaf_size, metric

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
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)

        # new logic
        neighbours = tree.query_radius(X=X, r=self.eps)
        G = {}
        cores = []
        for i, n in enumerate(neighbours):
            if len(n) >= self.min_samples:
                G[i] = n
                cores.append(i)
            else:
                G[i] = None

        from collections import deque

        color = -1
        colors = np.asarray([-1] * len(X))
        for c in cores:
            if colors[c] == -1:
                color += 1
                q = deque()
                q.append(c)

                while q:
                    v = q.popleft()
                    colors[v] = color

                    for u in G[v]:
                        if colors[u] == -1:
                            if G[u] is not None:
                                colors[u] = color
                                q.append(u)
                            else:
                                colors[u] = color
        labels = colors
        return labels


X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_2, true_labels_2 = make_moons(400, noise=0.075)

# init values
leaf_size = 40
metric = ['euclidean', 'manhattan', 'chebyshev']

eps = 0.2
min_samples = 5

# build tree
X = X_2
tree = KDTree(X, leaf_size=leaf_size, metric=metric[2])

dbscan = DBScan()
labels = dbscan.fit_predict(X=X_1, y=None)
print(labels)
