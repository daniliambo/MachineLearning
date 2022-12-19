from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def visualize_clusters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    print(unique_labels, unique_colors)
    colors = [unique_colors[l - 1] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


# tests
X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
X_2, true_labels_2 = make_moons(400, noise=0.075)

# init values
leaf_size = 40
metric = ['euclidean', 'manhattan', 'chebyshev']

eps = 0.2
min_samples = 5

# build tree
X = X_2
tree = KDTree(X, leaf_size=leaf_size, metric=metric[2])

# build Graph
G = dict()
for i in range(len(X)):
    G[i] = None  # keep indices in vertices

# fill the graph
for i in range(len(X)):
    query_result = tree.query_radius(X=X[i].reshape(1, -1), r=eps)[0]
    if len(query_result) >= min_samples:
        G[i] = query_result

# color the graph
from collections import deque

# find cores
cores = list()
for k, v in G.items():
    if v is not None:
        cores.append(k)

color = 0
colors = np.asarray([0] * len(X))
for c in cores:
    if colors[c] == 0:
        color += 1
        q = deque()
        q.append(c)

        while q:
            v = q.popleft()
            colors[v] = color

            for u in G[v]:
                if colors[u] == 0:
                    if G[u] is not None:
                        colors[u] = color
                        q.append(u)
                    else:
                        colors[u] = color
print(colors)
uniques, counts = np.unique(colors, return_counts=True)
print(uniques, counts)
labels = colors
visualize_clusters(X, labels)
