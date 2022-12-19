import numpy as np
import matplotlib.pyplot as plt

# out
n = 20
m = 4
X = np.random.randint(0, 10, size=(n, m))

n_clusters = 4
max_iter = 10

# in
res = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=X)
res = list(zip(res[0], res[1]))

next = np.random.randint(0, m, size=(n,))
prev = np.random.randint(0, m, size=(n,))
x = n_clusters - 1


def calculate_distance():
    return np.apply_along_axis(
        lambda point: min([(np.linalg.norm(point - centroid[0]), centroid[1]) for centroid in centroids]), axis=1,
        arr=X)


for _ in range(max_iter):
    while x < n_clusters:
        centroids = [[np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=res), _] for _ in
                     range(n_clusters)]

        closest_centroid = calculate_distance()
        x = len(np.unique(closest_centroid[:, 1]))

    closest_centroid = calculate_distance()

    print(closest_centroid[:, 1])

    next = closest_centroid[:, 1]
    if np.allclose(next, prev):
        break
    prev = closest_centroid[:, 1]

    for i in range(len(centroids)):
        closest_indices = list()
        pass
        for j in range(len(closest_centroid)):
            if closest_centroid[j][1] == i:
                closest_indices.append(j)

        centroids[i][0] = np.apply_along_axis(lambda x: np.mean(x), axis=0,
                                              arr=X[closest_indices])
