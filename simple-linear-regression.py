from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
import random

# dummy data
xs = np.array([1,2,3,4,5,6])
ys = np.array([11,19,33,39,52,61])

# crate random dataset
def create_dataset(amount, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(amount):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        if correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs), np.array(ys)

xs, ys = create_dataset(50, 30, correlation='pos')

# calculating m and b as per the algorithm
def best_fit_slope_y_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs * ys) ) /
          ( (mean(xs) * mean(xs)) - mean(xs * xs) ) )
    b = mean(ys) - m * mean(xs)
    return m, b

m, b = best_fit_slope_y_intercept(xs, ys)

# getting the regression line by using all of the x to 'y = mx + c'
regression_line = [(m*x)+b for x in xs]

# predicting a value
x_predict = 8
y_predict = m*x_predict + b
print(y_predict)

# calculating the accuracy
# for calculating the sum of squared error
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

# for calculating the r squared theory
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    se_regression_line = squared_error(ys_orig, ys_line)
    se_mean_line = squared_error(ys_orig, y_mean_line)
    return (1 - (se_regression_line / se_mean_line))

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# plotting the points and regression line
plt.scatter(xs, ys, marker='.')
plt.plot(xs, regression_line)
plt.scatter(x_predict,y_predict, color='red', marker='x')
plt.show()