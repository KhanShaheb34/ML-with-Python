import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create a dummy dataset
X = np.array([[1, 2],
              [2, 3],
              [3, 1],
              [6, 7],
              [9, 8],
              [8, 7]])

# Creating the classifier
clf = KMeans(n_clusters=2)
clf.fit(X)
centers = clf.cluster_centers_
labels = clf.labels_

colors = ["green", "red"]

# Visualize
for i in range(6):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]])
for c in centers:
    plt.scatter(c[0], c[1], marker="x")
plt.show()