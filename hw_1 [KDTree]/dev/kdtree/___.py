import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
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
    df = pandas.read_csv(path_to_csv)
    df = df.sample(frac=1)

    x = df.drop('label', axis=1).to_numpy()
    y = df['label']
    y = y.replace(["M", "B"], [1, 0]).to_numpy()
    return (x, y)


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
    df = pandas.read_csv(path_to_csv)
    df = df.sample(frac=1)

    x = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()

    return (x, y)


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
    size = X.shape[0]
    X_train = X[:int(size * ratio)]
    x_test = X[int(size * ratio):]
    y_train = y[:int(size * ratio)]
    y_test = y[int(size * ratio):]

    return X_train, y_train, x_test, y_test


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

    n_classes = len(np.unique(list(y_pred) + list(y_true)))
    N = len(y_true)
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    relevant = np.zeros(n_classes)

    for i in range(N):
        if y_pred[i] == y_true[i]:
            tp[y_true[i]] += 1
        else:
            fp[y_pred[i]] += 1
        relevant[y_true[i]] += 1

    precision = tp / (tp + fp)
    recall = tp / relevant
    accuracy = tp.sum() / N

    return precision, recall, accuracy


# Task 4


class Node:
    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def nearest_neighbors(self, point, k):
        pass


class BasicNode(Node):
    def __init__(self, left, right, median: float, feature_index: int):
        self.left = left
        self.right = right
        self.median = median
        self.feature_index = feature_index

    def merge(self, neighbors1, neighbors2):
        array1, dist1 = neighbors1
        array2, dist2 = neighbors2
        N, M = len(array1), len(array2)
        distances, indexes = np.zeros(N + M), np.zeros(N + M, dtype=int)
        i, j = 0, 0

        for current_index in range(N + M):
            if i < N and j < M:
                if dist1[i] <= dist2[j]:
                    distances[current_index] = dist1[i]
                    indexes[current_index] = array1[i]
                    i += 1
                else:
                    distances[current_index] = dist2[j]
                    indexes[current_index] = array2[j]
                    j += 1
            else:
                if j >= M:
                    distances[current_index] = dist1[i]
                    indexes[current_index] = array1[i]
                    i += 1
                else:
                    distances[current_index] = dist2[j]
                    indexes[current_index] = array2[j]
                    j += 1

        return indexes, distances

    def dist_to_hyperplance(self, point):
        return np.abs(point[self.feature_index] - self.median)

    def nearest_neighbors(self, point, k):
        if point[self.feature_index] < self.median:
            nearest_neighbors, distances = self.left.nearest_neighbors(point, k)
            to_visit = self.right
        else:
            nearest_neighbors, distances = self.right.nearest_neighbors(point, k)
            to_visit = self.left

        highest_distance = distances[-1]
        rest_nearest_neighbors, other_distances = np.array([]), np.array([])
        if (highest_distance > self.dist_to_hyperplance(point)) or (len(nearest_neighbors) < k):
            rest_nearest_neighbors, other_distances = to_visit.nearest_neighbors(point, k)

        merged_neighbors, merged_distances = self.merge((nearest_neighbors, distances),
                                                        (rest_nearest_neighbors, other_distances))

        return merged_neighbors[:k], merged_distances[:k]


class LeafNode(Node):
    def __init__(self, X: np.array, indices: np.array):
        self.X = X
        self.indices = indices

    def nearest_neighbors(self, point: np.array, k: int = 1) -> Tuple[np.array, np.array]:
        leaf_elements = self.X[self.indices]
        distances = np.apply_along_axis(lambda x: self.distance(point, x), 1, leaf_elements)
        sorted_k_indices = distances.argsort()[:k]
        neighbors = self.indices[sorted_k_indices]
        distances = distances[sorted_k_indices]

        return neighbors, distances


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
        self.leaf_size = leaf_size
        n, m = X.shape
        self.m = m
        self.root = self.build_tree(np.arange(n), 0)

    def build_tree(self, indices: np.array, feature_index: int) -> Node:
        data = self.X[indices]
        for _ in range(int(np.log2(self.m))):
            feature_values = data[:, feature_index]
            median = np.median(feature_values)
            less_indices = np.where(data[:, feature_index] < median)[0]
            greater_indices = np.where(data[:, feature_index] >= median)[0]

            if len(less_indices) >= self.leaf_size and len(greater_indices) >= self.leaf_size:
                left = self.build_tree(indices[less_indices], (feature_index + 1) % self.m)
                right = self.build_tree(indices[greater_indices], (feature_index + 1) % self.m)
                return BasicNode(left, right, median, feature_index)

            feature_index = (feature_index + 1) % self.m
        return LeafNode(self.X, indices)

    def query(self, X: np.array, k: int = 1) -> List[List]:
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
        get_neighbours = lambda point: self.root.nearest_neighbors(point, k)[0]
        result = np.apply_along_axis(get_neighbours, 1, X)

        return result


# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
