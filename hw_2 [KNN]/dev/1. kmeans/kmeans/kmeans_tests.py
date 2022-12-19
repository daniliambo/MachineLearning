# random init
# sample init
# k++

# random init
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def visualize_clusters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[l] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


def clusters_statistics(flatten_image, cluster_colors, cluster_labels):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    for remove_color in range(3):
        axes_pair = axes[remove_color]
        first_color = 0 if remove_color != 0 else 2
        second_color = 1 if remove_color != 1 else 2
        axes_pair[0].scatter([p[first_color] for p in flatten_image], [p[second_color] for p in flatten_image],
                             c=flatten_image, marker='.')
        axes_pair[1].scatter([p[first_color] for p in flatten_image], [p[second_color] for p in flatten_image],
                             c=[cluster_colors[c] for c in cluster_labels], marker='.')
        for a in axes_pair:
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
    plt.show()


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
            self.fit_random(X, y)
        elif self.init == "sample":
            self.fit_sample(X, y)
        elif self.init == "k-means++":
            self.fit_kmeans(X, y)

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


from sklearn.datasets import make_blobs, make_moons

X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
# visualize_clusters(X_1, true_labels_1)
X_2, true_labels_2 = make_moons(400, noise=0.075)
# visualize_clusters(X_2, true_labels_2)
# print(X_1)

n_clusters = 4
# max_iter = 200
# init = "random"
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_1)
# init random points
res = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=kmeans.X)
res = list(zip(res[0], res[1]))
points = [np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=res) for _ in
          range(kmeans.n_clusters)]

labels = kmeans.predict(X_1)
print(labels)
