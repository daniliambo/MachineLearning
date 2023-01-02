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

# build dist matrix
# think through the logic
distl = np.array([[[]] * (len(X) - i - 1) for i in range(len(X) - 1)], dtype=object)
# print(distl)

for i in range(len(X) - 1):  # чье
    for j in range(len(X) - i - 1):  # с кем
        distl[i][j] = np.append(distl[i][j],
                                np.array([np.linalg.norm(X[i] - X[1 + i + j]), [i, 1 + i + j]],
                                         dtype=object))
# flatten
# print(distl)
# find min
distances_prev = [item for sublist in distl for item in sublist]
distances_prev.sort(key=lambda x: x[0])
print(distances_prev)

# merge
merged_indices = []
distances_next = []
i = 0
while len(merged_indices) < len(X) - 1:
    print(distances_prev[i][1])
    if any(item in distances_prev[i][1] for item in merged_indices):
        pass
    else:
        merged_indices.extend(distances_prev[i][1])
        distances_next.append(distances_prev[i])
    i += 1
print(distances_next)
