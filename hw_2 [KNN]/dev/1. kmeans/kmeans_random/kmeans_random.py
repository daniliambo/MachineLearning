import numpy as np
import matplotlib.pyplot as plt

# init
np.random.seed(40)
n = 20
m = 4
X = np.random.randint(0, 10, size=(n, m))

res = np.apply_along_axis(lambda y: (min(y), max(y)), axis=0, arr=X)
res = list(zip(res[0], res[1]))

# print(X, 'X')
# print(res, 'res')
# print(centroids, 'centroids')
# calculate kmeans
n_clusters = 4
max_iter = 10
next = np.random.randint(0, m, size=(n,))
prev = np.random.randint(0, m, size=(n,))
x = n_clusters - 1


def calculate_distance():
    return np.apply_along_axis(
        lambda point: min([(np.linalg.norm(point - centroid[0]), centroid[1]) for centroid in centroids]), axis=1,
        arr=X)


for _ in range(max_iter):
    # для всех точек выборки найти ближайшую центроиду
    while x < n_clusters:
        centroids = [[np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=1, arr=res), _] for _ in
                     range(n_clusters)]

        closest_centroid = calculate_distance()
        x = len(np.unique(closest_centroid[:, 1]))

    # duplicate
    closest_centroid = calculate_distance()
    # print(x, n_clusters)

    # print(closest_centroid)
    # calculate all close for centroid movements
    next = closest_centroid[:, 1]
    # print(next.shape)
    print(np.allclose(next, prev))
    # if np.allclose(next, prev):
    #     break
    prev = closest_centroid[:, 1]

    # сдвинуть центр центроид относительно центра масс точек, затем посчитать опять расстояния и изменить центроиды
    for i in range(len(centroids)):
        closest_indices = list()
        pass
        # выбрать все семплы, которые лежат в этой центроиде
        for j in range(len(closest_centroid)):
            if closest_centroid[j][1] == i:
                closest_indices.append(j)

        # print(i, closes_indices)
        # reweight weights
        # print(X[closes_indices])
        centroids[i][0] = np.apply_along_axis(lambda x: np.mean(x), axis=0,
                                              arr=X[closest_indices])  # takes in the whole axis
        # print('-----')
        # print(i, centroids[i][0])

    # visualizing
    # print(centroids)
    # for _ in range(n):
    #     plt.scatter(x=X[i][0], y=X[i][1])
    # [plt.scatter(x=centorid[0][0], y=centorid[0][1], color='red') for centorid in centroids]
    # plt.show()
    # don't forget to reinit

# recalculate
