import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List
import pandas as pd


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    dataset = pd.read_csv(path_to_csv)
    labels = dataset['label'].replace(['M', 'B'], [1, 0]).to_numpy()
    features = dataset.drop(columns=['label']).to_numpy()
    return features, labels


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    dataset = pd.read_csv(path_to_csv)
    labels = dataset['label'].to_numpy()
    features = dataset.drop(columns=['label']).to_numpy()
    return features, labels


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    seed = np.random.RandomState(np.random.randint(0, 2 ** 16 - 1))
    train_sz = int(len(y) * ratio)
    seed.shuffle(X)
    seed.shuffle(y)
    X_train, y_train = X[:train_sz], y[:train_sz]
    X_test, y_test = X[train_sz:], y[train_sz:]
    return X_train, y_train, X_test, y_test


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    num_classes = np.unique(y_true).shape[0]
    precision, recall = np.empty(shape=num_classes), np.empty(shape=num_classes)
    for i in range(num_classes):
        precision[i] = np.shape(np.where((y_pred == i) & (y_true == y_pred)))[1] \
                       / np.shape(np.where((y_pred == i)))[1]
        recall[i] = np.shape(np.where((y_pred == i) & (y_true == y_pred)))[1] \
                    / np.shape(np.where((y_true == i)))[1]
    accuracy = np.shape(np.where((y_true == y_pred)))[1] / y_true.shape[0]
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
            merged = self.merge(x, k, neighbours, other_neighbours, distances, other_distances)
            print(merged)
            return merged
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

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        self.kd_train = KDTree(X, self.leaf_size)
        self.labels = y
        self.n_classes = len(np.unique(y))

    def predict_eval(self, nn: np.array) -> np.array:
        unique, counts = np.unique(nn, return_counts=True)
        prob_vect = np.zeros((self.n_classes,))
        for i in range(len(unique)):
            prob_vect[unique[i]] = counts[i]
        return prob_vect / self.n_classes

    def predict_proba(self, X: np.array) -> List[np.array]:
        nn = self.kd_train.query(X, self.n_neighbors)
        labels_nn = np.apply_along_axis(lambda x: self.labels[x], axis=1, arr=nn)
        return np.apply_along_axis(lambda x: self.predict_eval(nn=x), axis=1, arr=labels_nn)

    def predict(self, X: np.array) -> np.array:
        return np.argmax(self.predict_proba(X), axis=1)
