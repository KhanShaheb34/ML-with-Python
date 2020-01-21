import numpy as np 
import warnings
import matplotlib.pyplot as plt 
from matplotlib import style
from math import sqrt
from collections import Counter

# plot style
style.use("fivethirtyeight")

# dataset
dataset = {'r': [[1, 3], [4, 2], [3, 3]], 'b': [[8, 6], [7, 5], [7, 6]]}
new_feature = [4, 4]

# calculating K Nearest Neighbors
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("The dimension of your data is not smaller than K")

    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt( np.sum( (np.array(features) - np.array(predict))**2 ) )
            euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) )
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

group = k_nearest_neighbors(dataset, new_feature)


# plotting
[ [ plt.scatter(ii[0], ii[1], s=100, c=i) for ii in dataset[i] ] for i in dataset ]
plt.scatter(new_feature[0], new_feature[1], s=100, marker='x', c=group)
plt.show()