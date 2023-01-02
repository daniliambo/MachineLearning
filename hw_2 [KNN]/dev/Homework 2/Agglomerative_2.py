from sklearn.datasets import make_blobs, make_moons
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def visualize_clusters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[l] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


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
        if linkage == 'average':
            self.calc_dist = self.average_link
        elif linkage == 'single':
            self.calc_dist = self.single_link
        elif linkage == 'complete':
            self.calc_dist = self.complete_link

    def average_link(self, a, b):
        return np.nanmean([a, b], axis=0)

    def single_link(self, a, b):
        return np.where(a < b, a, b)

    def complete_link(self, a, b):
        return np.where(a >= b, a, b)

    def _calc_dist_mat(self, X, Y):
        dist_mat = (X ** 2).sum(-1).reshape(-1, 1) + (Y ** 2).sum(-1) - 2 * X @ Y.T  # l2
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
        idx = np.unravel_index(np.nanargmin(dist_mat, axis=None), dist_mat.shape)  # genius
        return idx

    # numpy.nanmean(arr, axis=None, dtype=None, out=None, keepdims=)) # get means ignoring nans

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


X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_2, true_labels_2 = make_moons(400, noise=0.075)

X = X_1
# print(X)
ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels = ac.fit_predict(X)
visualize_clusters(X, labels)