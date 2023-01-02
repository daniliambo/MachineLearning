from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(42)


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

X = X_1


# Task 3

class AgglomertiveClustering:
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
        self.n_clusters, self.linkage = n_clusters, linkage

    def average_linkage(self, a, b):
        return np.nanmean([a, b], axis=0)

    def complete_linkage(self, a, b):
        return np.where(a >= b, a, b)

    def single_linkage(self, a, b):
        return np.where(a < b, a, b)

    def _compute_dist_matrix(self, X):
        dist_mat = np.apply_along_axis(lambda el: np.apply_along_axis(lambda x: np.linalg.norm(el - x), axis=1, arr=X),
                                       axis=1, arr=X)
        np.fill_diagonal(dist_mat, np.nan)
        return dist_mat

    def reassign_labels(self, labels):
        unique = np.unique(labels)
        d = dict(zip(unique, np.arange(unique.shape[0])))
        labels = list(map(lambda x: d[x], labels))
        return labels

    def get_idx(self, dist_mat):
        return np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)

    def update_distance(self, dist_mat, idx):
        i, j = idx
        dists = self.link(dist_mat[i, :], dist_mat[j, :])
        dist_mat[i, :], dist_mat[:, i] = dists, dists
        dist_mat[j, :], dist_mat[:, j] = np.nan, np.nan
        np.fill_diagonal(dist_mat, np.nan)
        return dist_mat

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

        if self.linkage == 'average':
            self.link = self.average_linkage
        elif self.linkage == 'complete':
            self.link = self.complete_linkage
        elif self.linkage == 'single':
            self.link = self.single_linkage

        dist_mat = self._compute_dist_matrix(X)
        labels = np.arange(len(X))
        c = 0

        while c < len(X) - self.n_clusters:
            idx = self.get_idx(dist_mat)
            dist_mat = self.update_distance(dist_mat, idx)
            i, j = idx
            labels[labels == j] = i
            c += 1
        return self.reassign_labels(labels)


agg = AgglomertiveClustering(n_clusters=3, linkage='average')
labels = agg.fit_predict(X)

visualize_clusters(X, labels)
