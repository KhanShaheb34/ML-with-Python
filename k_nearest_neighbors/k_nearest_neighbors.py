import numpy as np 
import pandas as pd 
import warnings
import random
from math import sqrt
from collections import Counter

# calculating K Nearest Neighbors
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("The dimension of your data is not smaller than K")

    distances = []
    for group in data:
        for features in data[group]:
            # this is a high level version of calculating the euclidean distance
            euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) )
            distances.append([euclidean_distance, group])

    # finding the closest k points
    votes = [i[1] for i in sorted(distances)[:k]]
    # finding the most common group in the closest k points
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# reading the dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data')
# replacing the missing data
dataset.replace('?', -99999, inplace=True)
# dropping the id column
dataset.drop(['id'], 1, inplace=True)
# replacing the string data to float
dataset = dataset.astype(float).values.tolist()
# shuffling the dataset
random.shuffle(dataset)

# splitting the dataset for training and testing
test_size = 0.2
train_data = dataset[int(test_size*len(dataset)):]
test_data = dataset[:int(test_size*len(dataset))]

# grouping the data
train_set = { 2:[], 4:[] }
test_set = { 2:[], 4:[] }
for data in train_data:
    train_set[data[-1]].append(data[:-1])
for data in test_data:
    test_set[data[-1]].append(data[:-1])

# testing with test data and calculating accuracy
total = 0
correct = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data)
        if group == vote:
            correct += 1
        total += 1

accuracy = correct/total
print(accuracy)