import matplotlib.pyplot as plt 
from create_dataset import create_dataset
from best_fit_slope_y_intercept import best_fit_slope_y_intercept
from coefficient_of_determination import coefficient_of_determination

# crate random dataset
xs, ys = create_dataset(50, 30, correlation='pos')

# calculating m and b as per the algorithm
m, b = best_fit_slope_y_intercept(xs, ys)

# getting the regression line by using all of the x to 'y = mx + c'
regression_line = [(m*x)+b for x in xs]

# predicting a value
x_predict = 55
y_predict = m*x_predict + b
print(y_predict)

# calculating the accuracy
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# plotting the points and regression line
plt.scatter(xs, ys, marker='.')
plt.plot(xs, regression_line)
plt.scatter(x_predict,y_predict, color='red', marker='x')
plt.show()