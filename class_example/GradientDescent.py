import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """this class have to learn at main.py"""
    def __init__(self, learning_rate, max_iterations, threshold):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.threshold = threshold

    def fit(self, X, y):
        # Initialize weights to zeros
        self.theta = np.zeros(X.shape[1])
        # Keep track of cost for each iteration
        self.cost_history = []

        # Iterate until convergence or max_iterations is reached
        for i in range(self.max_iterations):
            # Calculate hypothesis and error
            h = np.dot(X, self.theta)
            error = h - y

            # Update weights using gradient descent
            gradient = np.dot(X.T, error) / len(X)
            self.theta -= self.learning_rate * gradient

            # Calculate cost and append to history
            cost = np.sum(error ** 2) / (2 * len(X))
            self.cost_history.append(cost)

            # Check for convergence
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.threshold:
                break

    def predict(self, X):
        return np.dot(X, self.theta)

    def plot_cost_history(self):
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()