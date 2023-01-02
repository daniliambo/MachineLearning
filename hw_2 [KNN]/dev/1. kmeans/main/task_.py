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


# Task 1

# fix random and kmeans++
class KMeans:
    def __init__(self, n_clusters: int, init="random", max_iter: int = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def calculate_distance(self, X):
        return np.apply_along_axis(
            lambda point: min([(np.linalg.norm(point - centroid[0]), centroid[1]) for centroid in self.centroids]),
            axis=1,
            arr=X)

    def predict(self, X: np.array) -> np.array:
        return np.array(list(map(int, self.calculate_distance(X)[:, 1])))

    def fit(self, X: np.array, y=None):
        self.X = X
        self.n, self.m = self.X.shape[0], self.X.shape[1]

        # init self.centroids
        if self.init == "random":
            self.fit_random(X, y)
        elif self.init == "sample":
            self.fit_sample(X, y)
        elif self.init == "k-means++":
            self.fit_kmeans(X, y)

        # fill self.centroids
        prev = np.random.randint(0, self.m, size=(self.n,))
        for _ in range(self.max_iter):

            closest_centroid = self.calculate_distance(X=self.X)

            # check for convergence
            next = closest_centroid[:, 1]
            if np.allclose(next, prev):
                break
            prev = next

            for i in range(len(self.centroids)):
                closest_indices = list()

                for j in range(len(closest_centroid)):
                    if closest_centroid[j][1] == i:
                        closest_indices.append(j)

                self.centroids[i][0] = np.apply_along_axis(lambda x: np.mean(x), axis=0,
                                                           arr=X[closest_indices])

                # visualization
                # [plt.scatter(x=centorid[0][0], y=centorid[0][1], color='red') for centorid in self.centroids]
                # plt.show()

    def fit_random(self, X, y=None):

        min_max_features = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=X)
        min_max_features = list(zip(min_max_features[0], min_max_features[1]))
        number_of_centroids_with_at_least_1_point = self.n_clusters - 1

        while number_of_centroids_with_at_least_1_point < self.n_clusters:
            self.centroids = [
                [np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=min_max_features), _]
                for _ in range(self.n_clusters)]

            closest_centroid = self.calculate_distance(X=self.X)
            number_of_centroids_with_at_least_1_point = len(np.unique(closest_centroid[:, 1]))

        # init centroids

    def fit_sample(self, X, y=None):

        indices_of_centroids = np.random.choice(self.n, size=self.n_clusters, replace=False)
        self.centroids = [[self.X[indices_of_centroids[_]], _] for _ in range(self.n_clusters)]

    def fit_kmeans(self, X, y=None):

        # init the first centorid
        indices = np.arange(self.n)
        index_of_centroid = np.random.choice(self.n, size=1)[0]
        self.centroids = []

        for i in range(self.n_clusters):
            self.centroids.append([self.X[index_of_centroid], i])
            indices = np.setdiff1d(indices, index_of_centroid)
            closest_centroids = self.calculate_distance(X=self.X[indices])

            dist_total = sum(list(map(lambda x: x ** 2, closest_centroids[:, 0])))
            dist_prob = list(map(lambda x: x ** 2 / dist_total, closest_centroids[:, 0]))
            index_of_centroid = np.random.choice(indices, size=1, p=dist_prob)[0]


# # tests
# X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
# # visualize_clusters(X_1, true_labels_1)
# X_2, true_labels_2 = make_moons(400, noise=0.075)
# # visualize_clusters(X_2, true_labels_2)
# #
#
# print(X_2[:5])
#
# n_clusters = 5
# max_iter = 200
# init = "sample"
#
# kmeans = KMeans(n_clusters=5, init=init)
# kmeans.fit(X_1[:5])
# labels = kmeans.predict(X_1[:5])
# print(labels)
