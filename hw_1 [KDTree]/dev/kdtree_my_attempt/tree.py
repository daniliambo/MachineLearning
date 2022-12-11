import numpy as np


class Node:

    def __init__(self, left, right, median, f) -> None:
        self.left_tree = left
        self.right_tree = right
        self.median = median
        self.index = f


class Leaf:

    def __init__(self, indices) -> None:
        self.indexes = indices


class KDTree:

    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.X = X
        self.size = leaf_size
        self.m = X.shape[1]
        self.root = self.make_tree(np.arange(X.shape[0]), 0)

    def merge(self, x: np.array, k: int, nearest_neighbours_1: np.array, nearest_neighbours_2: np.array,
              distances_1: np.array, distances_2: np.array):
        ln1, ln2 = len(nearest_neighbours_1), len(nearest_neighbours_2)
        i, j = 0, 0
        mereged_neighbours = np.empty((min(ln1 + ln2, k),), dtype=int)
        distances = np.empty((min(ln1 + ln2, k),), dtype=float)
        while i < ln1 and j < ln2 and i + j < k:
            if distances_1[i] < distances_2[j]:
                mereged_neighbours[i + j] = nearest_neighbours_1[i]
                distances[i + j] = distances_1[i]
                i += 1
            else:
                mereged_neighbours[i + j] = nearest_neighbours_2[j]
                distances[i + j] = distances_2[j]
                j += 1
        while i < ln1 and i + j < k:
            mereged_neighbours[i + j] = nearest_neighbours_1[i]
            distances[i + j] = distances_1[i]
            i += 1
        while j < ln2 and i + j < k:
            mereged_neighbours[i + j] = nearest_neighbours_2[j]
            distances[i + j] = distances_2[j]
            j += 1
        return mereged_neighbours, distances

    def make_tree(self, indices: np.array, feature: int):
        data = self.X[indices]
        median = np.median(data[:, feature])
        left_indices = np.where(data[:, feature] < median)[0]
        right_indices = np.where(data[:, feature] >= median)[0]
        if len(left_indices) >= self.size and len(right_indices) >= self.size:
            left = self.make_tree(indices[left_indices], (feature + 1) % self.m)
            right = self.make_tree(indices[right_indices], (feature + 1) % self.m)
            return Node(left, right, median, feature)
        else:
            return Leaf(indices)

    def nn(self, node, x: np.array, k: int = 1):
        if isinstance(node, Leaf):
            data = self.X[node.indexes]
            distances = np.apply_along_axis(lambda y: np.linalg.norm(x - y),
                                            axis=1, arr=data)
            indices = distances.argsort()[:k]
            return node.indexes[indices], distances[indices]
        else:
            if x[node.index] < node.median:
                neighbours, distances = self.nn(node.left_tree, x, k)
                to_visit = node.right_tree
            else:
                neighbours, distances = self.nn(node.right_tree, x, k)
                to_visit = node.left_tree
        radius = distances[-1]
        if radius > np.abs(node.median - x[node.index]) or len(neighbours) < k:
            other_neighbours, other_distances = self.nn(to_visit, x, k)
            return self.merge(x, k, neighbours, other_neighbours, distances, other_distances)
        else:
            return neighbours, distances

    def query(self, X: np.array, k: int = 1):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        return np.apply_along_axis(lambda y: self.nn(node=self.root, x=y, k=k)[0],
                                   axis=1, arr=X)  # берет по строкам, возвращает только индексы


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.kd_train = None
        self.labels = None
        self.n_classes = None

    def fit(self, X: np.array, y: np.array):
        self.kd_train = KDTree(X, self.leaf_size)
        self.labels = y
        self.n_classes = len(np.unique(y))

    def predict_eval(self, nn: np.array) -> np.array:  # takes in 1 sample and predicts classes
        unique, counts = np.unique(nn, return_counts=True)
        prob_vect = np.zeros((self.n_classes,))
        for i in range(len(unique)):
            prob_vect[unique[i]] = counts[i]
        return prob_vect

    def predict_proba(self, X: np.array):
        nn = self.kd_train.query(X, self.n_neighbors)  # returns indices of nns
        labels_nn = np.apply_along_axis(lambda x: self.labels[x], axis=1, arr=nn)  # returns labels of nns
        print(labels_nn, 'labels_nn')
        return np.apply_along_axis(lambda x: self.predict_eval(nn=x), axis=1, arr=labels_nn)

    # X_test = np.asarray([[1, 7], [5, 1]])
    # knn.predict_proba(X_test)
    # TypeError: only integer scalar arrays can be converted to a scalar index

    def predict(self, X: np.array) -> np.array:
        return np.argmax(self.predict_proba(X), axis=1)


# init
X = np.asarray([[6, 3], [7, 4], [6, 9], [2, 6], [7, 4], [3, 7], [7, 2], [5, 4], [1, 7], [5, 1]])
X_test = np.asarray([[1, 7], [5, 1]])
y = [round(np.random.random()) for i in range(len(X))]
print(type(X_test))
print('X', X)
print('X_test', X_test)
print('y', y)

# params
leaf_size = 3
k = 2

# fit predict
knn = KNearest(k, leaf_size)
knn.fit(X, y)

X_test = np.asarray([[1, 7], [5, 1]])
knn.predict_proba(X_test)
