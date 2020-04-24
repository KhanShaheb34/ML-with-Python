[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/KhanShaheb34/ML-with-Python) 

# ML-with-Python
Learning ML with Python from [sentdex's tutorial series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

## Simple Linear Regression (SLR)
In linear regression we have some x values and some y values, and the problem is to find the best fit line for the graph of x and y.

In simple word we have to find `y = mx + c` line. From an array of `Xs` and an array of `ys`. To find the line we have to find the slope `m` and the y-intercept `c`.

The way to find `m` is:

![SLR Slope](/Images/slr-slope.png)

And the way to find c is:

![SLR y-intercept](/Images/slr-y-intercept.png)

### Accuracy of the best fit line
The error of a line from a point is the square of the perpendicular distance between them. It is squared because the distance can be negative too.

The squared error (SE) of a line is the sum of all errors from every point of a graph to a line.

The squared error of the best fit line is expressed by: ![SE(best-fit-line)](/Images/se-bfl.png)

And the squared error of the mean line is expressed by: ![SE(mean-line)](/Images/se-meanline.png)

According to **R Squared Theory** the accuracy of a best fite line is:

![R-squared Theory](/Images/r-squared.png)

#### [Implimentation from scratch](/simple_linear_regression/Simple_Linear_Regression.ipynb)

## K Nearest Neighbors (KNN)
When a dataset is classified in some groups in a graph, the K Nearest Neighbors or KNN is used to determine the class of a new point in the graph. It determines the class by finding out the nearest K points in the graph, and calculating which class appears most of the times in those K points.

Ususally the distance is calculated by the distance of the two different point in a graph. As example Eucledian Algorithm.

According to the Eucledian Algorithm, the distance between two point in a graph is:

![Euclidean Distance](/Images/euclid_dist.png)

> It is same as the Euclidian Distance algorithm we read in school and collage where ![Two Dimensional Euclidean Distance](/Images/euclid_dist_simp.gif). Just it is expressed in a well mannner! 😛

#### [Implimentation from scratch](/k_nearest_neighbors/K_Nearest_Neighbors.ipynb)
#### [Example with Scikit Learn](/using_sklearn/k_nearest_neighbors/K_Nearest_Neighbors.ipynb)

## Support Vector Machine (SVM)
The objective of SVM is to find the best splitting boundary in data. It is also used to classify data.

> Couldn't understand how the f\*ck this algorithm works!

#### [Implimentation from scratch](/support_vector_machine/support_vector_machine.py)
#### [Example with Scikit Learn](/using_sklearn/support_vector_machine/Support_Vector_Machine.ipynb)


## K-Means CLustering
K-means Clustering is a unsupervised learning algorithm which creates `k` clusters among the dataset.
The center of cluster gets selected through iteration. 

First of all it selects `k` center points randomly then finds out the closest point from the center points. Then it finds out the mean distance of the closest points from each center and move the center to that mean. And do this reapitively untill the centers stop moving.

Now the center points are the center of the clusters.

#### [Example with Scikit Learn](/KMeans_Clustering/kmeans_sklearn.py)

## Mean-Shift Clustering
Mean-shift algorithm is also an unsupervised clustering algorithm. As like [K-means Clustering](#K-Means-Clustering) it finds out the centers of the cluster but this time the number of centers aren't defined at first.

This algorithm takes every point of the dataset as a center at first and for each point it finds other points in the radius of that point and then move the center to the mean of the points that appear in the radius. By doing this reapitedly the centers converges to some particular points. And that points are the centers of the clusters.
