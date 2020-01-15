from statistics import mean

# for calculating the sum of squared error
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

# for calculating the r squared theory
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    se_regression_line = squared_error(ys_orig, ys_line)
    se_mean_line = squared_error(ys_orig, y_mean_line)
    return (1 - (se_regression_line / se_mean_line))