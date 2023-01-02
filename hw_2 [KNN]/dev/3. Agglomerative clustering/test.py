from pprint import pprint as pp
import numpy as np
from sklearn.datasets import make_blobs, make_moons

np.random.seed(42)

a = np.array([1, 2, 3])
b = np.array([3, 4, 5])

print(np.nanmean([a, b], axis=0))

X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_2, true_labels_2 = make_moons(400, noise=0.075)

X = X_1[:5]


# pp(c)
# pp(X)


def _calc_dist_mat(arr):
    dist_mat = np.apply_along_axis(lambda el: np.apply_along_axis(lambda x: np.linalg.norm(el - x), axis=1, arr=arr),
                                   axis=1,
                                   arr=arr)
    return np.triu(dist_mat)


dist_mat = _calc_dist_mat(X)
print('dist_mat')
pp(dist_mat)


def remap_labels(
        labels):  # np.array [0 1 0 0 0 0 6 0 6 6 6 1 6 0 6 0 6 6 1 0] -> [0 1 0 0 0 0 2 0 2 2 2 1 2 0 2 0 2 2 1 0]
    unique = np.unique(labels)
    d = dict(zip(unique, np.arange(unique.shape[0])))
    labels = list(map(lambda x: d[x], labels))
    print(labels)


remap_labels(np.array([0, 1, 0, 0, 0, 0, 6, 0, 6, 6, 6, 1, 6, 0, 6, 0, 6, 6, 1, 0]))


def upd_dst(self, idx, dist_mat):  # upd_dst
    i, j = idx
    new_dst = self.calc_dist(dist_mat[i, :], dist_mat[j, :])
    dist_mat[i, :], dist_mat[:, i] = new_dst, new_dst
    dist_mat[j, :], dist_mat[:, j] = np.nan, np.nan
    np.fill_diagonal(dist_mat, np.nan)

    return dist_mat


print(X)
print(np.unravel_index(np.argmax(X), X.shape))

a, b = 2
print(a, b)
