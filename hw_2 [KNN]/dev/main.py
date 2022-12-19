from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn


# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """

        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.

        """

        self._init_method = init
        self.n_clust = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def _calc_dist_mat(self, X, Y):
        dist_mat = (X ** 2).sum(-1).reshape(-1, 1) + (Y ** 2).sum(-1) - 2 * X @ Y.T

        return dist_mat

    def sample_random(self, X):
        min_v, max_v = X.min(), X.max()

        return np.random.uniform(min_v, max_v, size=(self.n_clust, X.shape[1]))

    def sample_sample(self, X):
        cents_idx = np.random.choice(np.arange(X.shape[0]), self.n_clust, replace=False)

        return X[cents_idx]

    def sample_pp(self, X):
        cent = []
        idx = np.random.choice(np.arange(X.shape[0]), 1)
        cent.append(X[idx][0])

        for i in range(self.n_clust - 1):
            dst = self._calc_dist_mat(X, np.array(cent)).min(-1) ** 2
            sum_dst = dst.sum()

            idx = np.random.choice(np.arange(X.shape[0]), 1, p=dst / sum_dst)
            cent.append(X[idx][0])

        return np.array(cent)

    def _fit(self, X):
        if self._init_method == 'random':
            self.centroids = self.sample_random(X)
        elif self._init_method == 'sample':
            self.centroids = self.sample_sample(X)
        elif self._init_method == 'k-means++':
            self.centroids = self.sample_pp(X)

        prev_classes = np.full((X.shape[0],), -1)

        for it in range(self.max_iter):
            dist = self._calc_dist_mat(X, self.centroids)
            classes = np.argmin(dist, axis=-1)

            if (prev_classes == classes).all():
                break
            prev_classes = classes

            for c in range(self.n_clust):
                if len(X[classes == c]) > 0:
                    self.centroids[c] = X[classes == c].mean(axis=0)

        return classes

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """
        for i in range(5):
            classes = self._fit(X)

            if len(np.unique(classes)) == self.n_clust:
                break

        return classes

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).

        """
        return np.argmin(self._calc_dist_mat(X, self.centroids), axis=-1)


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
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

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

        labels = np.full((X.shape[0]), -1)
        c = 0
        neigh_in_rad = tree.query_radius(X, self.eps)

        for idx, el in enumerate(X):
            if labels[idx] != -1:
                continue

            neighs = deque(neigh_in_rad[idx])
            if len(neighs) < self.min_samples:
                labels[idx] = -1  # noise
                continue

            while len(neighs) > 0:
                id_neigh = neighs.pop()
                if labels[id_neigh] != -1 or idx == id_neigh:
                    continue
                if len(neigh_in_rad[id_neigh]) >= self.min_samples:
                    neighs.extend(neigh_in_rad[id_neigh])
                labels[id_neigh] = c

            labels[idx] = c
            c += 1

        return labels


# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
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
        if linkage == 'average':
            self.calc_dist = self.average_link
        elif linkage == 'single':
            self.calc_dist = self.single_link
        elif linkage == 'complete':
            self.calc_dist = self.complete_link

    def average_link(self, a, b):
        return np.nanmean([a, b], axis=0)

    def single_link(self, a, b):
        return np.where(a < b, a, b)  #

    def complete_link(self, a, b):
        return np.where(a >= b, a, b)

    def _calc_dist_mat(self, X, Y):
        dist_mat = (X ** 2).sum(-1).reshape(-1, 1) + (Y ** 2).sum(-1) - 2 * X @ Y.T
        np.fill_diagonal(dist_mat, np.nan)

        return dist_mat

    def remap_labels(self, labels):
        unique = np.unique(labels)
        real = np.arange(len(unique))
        d = dict(zip(unique, real))

        labels = np.vectorize(d.get)(labels)

        return labels

    def upd_dst(self, idx, dist_mat):
        i, j = idx
        new_dst = self.calc_dist(dist_mat[i, :], dist_mat[j, :])
        dist_mat[i, :], dist_mat[:, i] = new_dst, new_dst
        dist_mat[j, :], dist_mat[:, j] = np.nan, np.nan

        np.fill_diagonal(dist_mat, np.nan)

        return dist_mat

    def get_min_id(self, dist_mat):
        idx = np.unravel_index(np.nanargmin(dist_mat, axis=None), dist_mat.shape)

        return idx

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
        dist_mat = self._calc_dist_mat(X, X)
        labels = np.arange(len(X))
        c = 0

        while c < len(X) - self.n_clusters:
            i, j = self.get_min_id(dist_mat)
            labels[labels == j] = i
            dist_mat = self.upd_dst((i, j), dist_mat)
            c += 1

        return self.remap_labels(labels)
