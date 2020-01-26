import numpy as np 
import matplotlib.pyplot as plt 

class Support_Vector_Machine:
    # calling the constructor
    def __init__(self, visualization = True):
        self.visualization = visualization
        # for visualizing plots
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # training method
    def fit(self, data):
        self.data = data
        optimum_dict = {}

        transformations = [[1, 1],
                          [1, -1],
                          [-1, -1],
                          [-1, 1]]

        # finding the maximum and minimum feature in dataset
        all_data = []
        for group in self.data:
            for featureset in self.data[group]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # declaring the step size for convergence
        step_size = [self.max_feature_value * 0.1,
                     self.max_feature_value * 0.01,
                     self.max_feature_value * 0.001]

        b_range_multiplier = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        # stepping towards the optimim value
        for step in step_size:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False

            # running until it is optimized
            while not optimized:
                # checking for every possible b from negative to positive max feature value * b range multiplier with step of step * b_multiples
                for b in np.arange(-1*self.max_feature_value*b_range_multiplier,
                                    self.max_feature_value*b_range_multiplier,
                                    step*b_multiple):
                    # checking for every possible transformation of w
                    for transformation in transformations:
                        w_t = w*transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False  
                        if found_option:
                            optimum_dict[np.linalg.norm(w_t)] = [w_t, b]
            
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step
            
            norms = sorted([n for n in optimum_dict])
            optimum_choice = optimum_dict[norms[0]]
            self.w = optimum_choice[0]
            self.b = optimum_choice[1]
            latest_optimum = optimum_choice[0][0] + step * 2

    # for predicting a value
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification
    
    # visualizing data
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], c=self.colors[i]) for x in self.data[i]] for i in self.data]
        
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        hyp_x_min = self.min_feature_value * 0.9
        hyp_x_max = self.max_feature_value * 1.1

        # plotting positive support vector
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # plotting negative support vector
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # plotting positive support vector
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()

# the test data
data_dict = {
    -1: np.array([[1,7], [2,8], [3,8]]),
     1: np.array([[5,1], [6,-1], [7,3]])
}

# creating svm prototype
SVM = Support_Vector_Machine()
SVM.fit(data_dict)
SVM.visualize()
