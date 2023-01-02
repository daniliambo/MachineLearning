import pandas as pd
import numpy as np


def read_cancer_dataset(path_to_csv: str):
    dataset = pd.read_csv(path_to_csv)
    labels = dataset['label'].replace(['M', 'B'], [1, 0]).to_numpy()
    features = dataset.drop(columns=['label']).to_numpy()
    return features, labels


X, y = read_cancer_dataset('cancer.csv')
f = 0
indices = np.arange(0, X.shape[0], 1)
median = np.median(X[indices, f])
X1 = X[24:42]

ind1 = [13, 23]
ind2 = [1, 58]
dist1 = [432.432, 23.43]
dist2 = [0.432, 4302.32]

data = list(zip(ind1, dist1))
data.extend(zip(ind2, dist2))
data.sort(key=lambda x: x[1])
nearest_neighbours_indices, nearest_neighbours_distances = list(zip(*data))
print(nearest_neighbours_indices, nearest_neighbours_distances)

# np.where tests
X = np.asarray([[6, 3], [7, 4], [6, 9], [2, 6], [7, 4], [3, 7], [7, 2], [5, 4], [1, 7], [5, 1]])
print(X.shape)
median = np.apply_along_axis(np.median, axis=0, arr=X)
print(median)
print(X[:, 0])
ans = np.where(X[:, 0] < median[0])[0]
indices = np.arange(len(X))
print(indices[ans])

print(ans)
