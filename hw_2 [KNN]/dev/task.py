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
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def fit(self, X: np.array, y=None):
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
        self.X = X
        self.n, self.m = self.X.shape[0], self.X.shape[1]

        if self.init == "random":
            self.fit_random(X)
        elif self.init == "sample":
            self.fit_sample(X)
        elif self.init == "k-means++":
            self.fit_kmeans(X)

    def calculate_distance(self, X):
        return np.apply_along_axis(
            lambda point: min([(np.linalg.norm(point - centroid[0]), centroid[1]) for centroid in self.centroids]),
            axis=1,
            arr=X)

    def fit_random(self, X, y=None):
        # init
        res = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=X)
        res = list(zip(res[0], res[1]))

        prev = np.random.randint(0, self.m, size=(self.n,))
        x = self.n_clusters - 1

        for _ in range(self.max_iter):
            while x < self.n_clusters:
                self.centroids = [[np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=res), _] for
                                  _ in
                                  range(self.n_clusters)]

                closest_centroid = self.calculate_distance(X=self.X)
                x = len(np.unique(closest_centroid[:, 1]))

            closest_centroid = self.calculate_distance(X=self.X)

            next = closest_centroid[:, 1]
            if np.allclose(next, prev):
                break
            prev = closest_centroid[:, 1]

            for i in range(len(self.centroids)):
                closest_indices = list()
                pass
                for j in range(len(closest_centroid)):
                    if closest_centroid[j][1] == i:
                        closest_indices.append(j)

                self.centroids[i][0] = np.apply_along_axis(lambda x: np.mean(x), axis=0,
                                                           arr=X[closest_indices])

            # visualization
            # [plt.scatter(x=centorid[0][0], y=centorid[0][1], color='red') for centorid in self.centroids]
            # plt.show()

    def fit_sample(self, X, y=None):
        pass

    def fit_kmeans(self, X, y=None):
        pass

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
        return list(map(int, self.calculate_distance(X)[:, 1]))


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
        pass

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
        pass


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
        pass

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
        pass


X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
# visualize_clusters(X_1, true_labels_1)
X_2, true_labels_2 = make_moons(400, noise=0.075)
# visualize_clusters(X_2, true_labels_2)
# print(X_1)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_1)
print(X_1)
labels = kmeans.predict(X_1)
print(labels)

# n_clusters = 4
# max_iter = 200
# init = "random"
# kmeans = KMeans(n_clusters=n_clusters)
# kmeans.fit(X_1)
# labels = kmeans.predict(X_1)
# print(labels)

# init random points
# res = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=kmeans.X)
# res = list(zip(res[0], res[1]))
# points = [np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=res) for _ in
#           range(kmeans.n_clusters)]

