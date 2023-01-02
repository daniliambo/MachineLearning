import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """

    df = pd.read_csv(path_to_csv)

    def preprocess_data(_):
        df = _.copy()
        df = df.sample(frac=1.0, random_state=42)

        # define y
        y = df.pop('label')
        d = {'M': 1, 'B': 0}
        y = y.replace(d, inplace=False)  # replace on Series
        # y = y.rename({'M'}) # rename with cols for example
        y = np.array(y)

        # define X
        X = df.to_numpy()
        return X, y

    X, y = preprocess_data(df)

    return (X, y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.

    """

    df = pd.read_csv(path_to_csv)

    def preprocess_data(_):
        df = _.copy()
        df = df.sample(frac=1.0, random_state=42)

        # define y
        y = df.pop('label')
        y = np.array(y)

        # define X
        X = df.to_numpy()
        return X, y

    X, y = preprocess_data(df)

    return (X, y)


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.
    """

    def preprocess_data_tts(X, y, ratio):
        N = len(X)
        split_index = int(N * ratio)
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_test = X[split_index:]
        y_test = y[split_index:]

        return (X_train, y_train, X_test, y_test)

    X_train, y_train, X_test, y_test = preprocess_data_tts(X, y, ratio)
    return (X_train, y_train, X_test, y_test)


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """

    def preprocess_inputs(y_pred, y_true):
        N = len(y_pred)

        classes_true = np.unique(y_pred)
        precision, recall, accuracy = list(), list(), list()
        N_classes = len(classes_true)
        tps = 0

        for i in range(N_classes):
            class_ = classes_true[i]

            tp = sum([1 for j, x in enumerate(y_pred) if x == class_ and y_true[j] == class_])
            fp = sum([1 for j, x in enumerate(y_pred) if x == class_ and y_true[j] != class_])
            fn = sum([1 for j, x in enumerate(y_pred) if x != class_ and y_true[j] == class_])

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)

            tps += tp

            precision = np.append(precision, prec)
            recall = np.append(recall, rec)

        accuracy = tps / N
        return (precision, recall, accuracy)

    precision, recall, accuracy = preprocess_inputs(y_pred, y_true)
    return precision, recall, accuracy


# Task 4

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
                                   axis=1, arr=X)


# Task 5

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

    def predict_eval(self, nn: np.array) -> np.array:
        unique, counts = np.unique(nn, return_counts=True)
        prob_vect = np.zeros((self.n_classes,))
        for i in range(len(unique)):
            prob_vect[unique[i]] = counts[i]
        return prob_vect

    def predict_proba(self, X: np.array):
        nn = self.kd_train.query(X, self.n_neighbors)
        labels_nn = np.apply_along_axis(lambda x: self.labels[x], axis=1, arr=nn)
        return np.apply_along_axis(lambda x: self.predict_eval(nn=x), axis=1, arr=labels_nn)

    def predict(self, X: np.array) -> np.array:
        return np.argmax(self.predict_proba(X), axis=1)


# X = np.asarray([[1, 3], [2, 1], [3, 1], [2, 6], [7, 4], [3, 7], [7, 2], [5, 4], [1, 7], [5, 1]])
# y = np.asarray([2, 3, 0, 2, 2, 3, 0, 0, 2, 1])
# points = np.asarray([[4, 7], [4, 2], [2, 3], [2, 1]])
# leaf_size = 2
# k = 3
# knn = KNearest(n_neighbors=3, leaf_size=5)
# knn.fit(X, y=y)
# probas = knn.predict_proba(points)
# print(probas)
# ans = knn.predict(points)
# print(ans)

X, y = read_cancer_dataset('cancer.csv')
X1 = X[24:26]  # 18
knn = KNearest(n_neighbors=3, leaf_size=5)
knn.fit(X, y=y)
probas = knn.predict_proba(X1)
ans = knn.predict(X1)
print(probas, ans)
