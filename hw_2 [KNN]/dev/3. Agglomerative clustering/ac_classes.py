from sklearn.datasets import make_blobs, make_moons
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


# tests
X_1, true_labels_1 = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
# visualize_clusters(X_1[:50, ], true_labels_1[:50])
X_2, true_labels_2 = make_moons(400, noise=0.075)
# visualize_clusters(X_2[:50, ], true_labels_2[:50])

X = X_1[:5]

