from statistics import mean

# calculating m and b as per the algorithm
def best_fit_slope_y_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs * ys) ) /
          ( (mean(xs) * mean(xs)) - mean(xs * xs) ) )
    b = mean(ys) - m * mean(xs)
    return m, b