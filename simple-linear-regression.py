from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 

# loading data 
xs = np.array([1,2,3,4,5,6])
ys = np.array([11,19,33,39,52,61])

# calculating m and b as per the algorithm
def best_fit_slope_y_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs * ys) ) /
          ( (mean(xs) * mean(xs)) - mean(xs * xs) ) )
    b = mean(ys) - m * mean(xs)
    return m, b

m, b = best_fit_slope_y_intercept(xs, ys)

# getting the regression line by using all of the x to 'y = mx + c'
regression_line = [(m*x)+b for x in xs]

# plotting the points and regression line
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()