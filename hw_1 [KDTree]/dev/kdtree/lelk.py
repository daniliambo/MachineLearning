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

    data = pandas.read_csv(path_to_csv)
    data_shuffled = data.sample(frac=1).reset_index()

    labels = data_shuffled['label'].replace(['M', 'B'], [1, 0]).to_numpy()

    for (columnName, columnData) in data_shuffled.iteritems():
        if columnName != 'label':
            data_shuffled[columnName] = columnData.values / np.max(columnData.values)

    data_shuffled = data_shuffled.drop('label', axis=1).drop('index', axis=1)
    features = data_shuffled.to_numpy()

    return (features, labels)


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
    data = pandas.read_csv(path_to_csv)
    data_shuffled = data.sample(frac=1).reset_index()

    labels = data_shuffled['label'].to_numpy()
    for (columnName, columnData) in data_shuffled.iteritems():
        if columnName != 'label':
            data_shuffled[columnName] = columnData.values / np.max(columnData.values)

    data_shuffled = data_shuffled.drop('label', axis=1).drop('index', axis=1)
    features = data_shuffled.to_numpy()
    return (features, labels)


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
    train_size = int(ratio * len(X)) + 1
    X_train, X_test = np.split(X, [train_size])
    y_train, y_test = np.split(y, [train_size])

    return X_train, y_train, X_test, y_test


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

    classes = np.sort(np.unique(list(y_true) + list(y_pred)))

    precision = []
    recall = []

    for el in classes:

        if len(np.where(y_pred == el)[0]) != 0:
            precision.append(len(np.where((y_pred == y_true) & (y_pred == el))[0]) / len(np.where(y_pred == el)[0]))
        else:
            precision.append(0)
        if len(np.where(y_true == el)[0]) != 0:
            recall.append(len(np.where((y_pred == y_true) & (y_pred == el))[0]) / len(np.where(y_true == el)[0]))
        else:
            recall.append(0)

    accuracy = len(np.where(y_true == y_pred)[0]) / len(y_true)

    return np.array(precision), np.array(recall), accuracy


# Task 4  
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

        def makeTree(points, k, i=0):

            if len(points) > leaf_size:

                median = np.median(points, axis=0)[i]

                left_points = points[points[:, i] < median]
                right_points = points[points[:, i] >= median]

                iprev = i
                if len(left_points) > 0 and len(right_points) > 0:
                    dist_for_left = np.min(np.abs(left_points[:, i] - median))
                    dist_for_right = np.min(np.abs(right_points[:, i] - median))

                    i = (i + 1) % k

                    return (
                        (makeTree(left_points, k, i), dist_for_left), (makeTree(right_points, k, i), dist_for_right),
                        (iprev, median))
                else:

                    i = (i + 1) % k
                    while i != iprev:

                        median = np.median(points, axis=0)[i]

                        left_points = points[points[:, i] < median]
                        right_points = points[points[:, i] >= median]

                        if len(left_points) > 0 and len(right_points) > 0:
                            dist_for_left = np.min(np.abs(left_points[:, i] - median))
                            dist_for_right = np.min(np.abs(right_points[:, i] - median))

                            i = (i + 1) % k

                            return ((makeTree(left_points, k, i), dist_for_left),
                                    (makeTree(right_points, k, i), dist_for_right), (iprev, median))

                        i = (i + 1) % k
                    return np.array([points])

            else:

                return np.array([points])

        def knn(node, point, k):

            if len(node) > 1:

                if point[node[2][0]] < node[2][1]:

                    neighb = knn(node[0][0], point, k)

                    max_distance = np.linalg.norm(neighb[-1] - point)

                    if len(neighb) < k or max_distance > node[1][1] + np.abs(point[node[2][0]] - node[2][1]):

                        neighb = np.concatenate((neighb, knn(node[1][0], point, k)))

                        return neighb[np.argsort(np.linalg.norm(neighb - point, axis=1))[:k]]

                    else:
                        return neighb

                else:
                    neighb = knn(node[1][0], point, k)
                    max_distance = np.linalg.norm(neighb[-1] - point)

                    if len(neighb) < k or max_distance > node[0][1] + np.abs(point[node[2][0]] - node[2][1]):
                        neighb = np.concatenate((neighb, knn(node[0][0], point, k)))

                        return neighb[np.argsort(np.linalg.norm(neighb - point, axis=1))[:k]]
                    else:
                        return neighb



            else:
                neighb = node[0]
                neigbours = neighb[np.argsort(np.linalg.norm(neighb - point, axis=1))]

                if len(neigbours) >= k:
                    return neigbours[:k]
                else:
                    return neigbours

        self._tree = makeTree(X, X.shape[1])
        self._indices = {tuple(el): i for i, el in enumerate(X)}
        self._knn = knn

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
        ans = []

        for el in X:
            ans.append([self._indices[tuple(el)] for el in self._knn(self._tree, el, k)])

        return ans


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
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self._tree = KDTree(X, self.leaf_size)
        self.y = y
        self.X = X

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

        ans = []

        def triangle_kernel(neighb, center, n, c=1):

            radius = np.linalg.norm(center - neighb[-1])
            if radius != 0:
                return np.exp(-c * np.linalg.norm(center - neighb[:-1], axis=1) / radius)

                # return 1 - np.linalg.norm(center - neighb[:-1] , axis = 1)/radius + 10**(-3)
            else:
                return np.full((n), 1)
            # return np.full((n), 1)

        neighbours = self._tree.query(X, self.n_neighbors + 1)

        classes = np.sort(np.unique(self.y))
        for i, el in enumerate(X):

            kern = triangle_kernel(self.X[neighbours[i]], el, self.n_neighbors)
            prob = []
            for clas in classes:
                prob.append(np.sum(kern[np.where(self.y[neighbours[i][:-1]] == clas)[0]]))

            sum = np.sum(prob)

            ans.append(np.array([el / sum for el in prob]))
            # ans.append(np.array([1 - np.count_nonzero(self.y[neighbours[i]])/self.n_neighbors,np.count_nonzero(self.y[neighbours[i]])/self.n_neighbors]))

        return ans

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
