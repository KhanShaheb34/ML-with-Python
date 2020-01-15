import random
import numpy as np 

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