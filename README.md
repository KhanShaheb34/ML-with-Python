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

#### [Implimentation](/simple_linear_regression/simple_linear_regression.py)

## K Nearest Neighbors (KNN)
When a dataset is classified in some groups in a graph, the K Nearest Neighbors or KNN is used to determine the class of a new point in the graph. It determines the class by finding out the nearest K points in the graph, and calculating which class appears most of the times in those K points.

Ususally the distance is calculated by the distance of the two different point in a graph. As example Eucledian Algorithm.

According to the Eucledian Algorithm, the distance between two point in a graph is:

![Euclidean Distance](/Images/euclid_dist.png)

> It is same as the Euclidian Distance algorithm we read in school and collage where ![Two Dimensional Euclidean Distance](/Images/euclid_dist_simp.gif). Just it is expressed in a well mannner! 😛

[Example with Scikit Learn](/using_sklearn/k_nearest_neighbors/K_Nearest_Neighbors.ipynb)