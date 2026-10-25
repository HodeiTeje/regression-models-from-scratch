import numpy as np


class LinearRegression:

    def __init__(self, learning_rate = 0.01, max_iterations = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        self.weights = None
        self.bias = 0.0
        self.tolerance = 1e-12

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.weights + self.bias

    def compute_cost(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        return np.mean((self.predict(X) - y)**2)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Get the number of features and instances
        n_features = X.shape[1] 
        n_instances = X.shape[0]

        # Initialize the weights and the bias with random values between 0 and 1
        self.weights = np.random.rand(n_features) 
        self.bias = np.random.rand()

        costs = []
        # Train the model using gradient descent
        for iteration in range(self.max_iterations):
            predictions = self.predict(X)
            errors = predictions - y

            costs.append(self.compute_cost(X,y))

            # Get the gradients of the parameters
            d_weights = (2/n_instances)*(X.T @ errors)
            d_bias = (2/n_instances)*np.sum(errors)

            # Update the parameters 
            self.weights = self.weights - self.learning_rate * d_weights
            self.bias = self.bias - self.learning_rate * d_bias

            # Stopping criterion
            gradient_norm = np.sqrt(np.sum(d_weights ** 2) + d_bias ** 2)
            if(gradient_norm < self.tolerance):
                break

        return costs



