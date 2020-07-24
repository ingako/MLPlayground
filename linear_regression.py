#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, alpha, num_params):
        self.alpha = alpha
        self.theta = np.zeros(num_params)

    def partial_fit(self, X, y, num_iters):
        return self._gradient_descent(X, y, num_iters)

    def predict(self, X):
        return np.dot(X, self.theta)

    def _gradient_descent(self, X, y, num_iters):
        m = y.shape[0]
        J_history = []

        for i in range(num_iters):
            self.theta = self.theta - self.alpha / m * (np.dot(X, self.theta) - y).dot(X)
            J_history.append(self._compute_cost(X, y))

        return self.theta, J_history

    def _compute_cost(self, X, y):
        m = y.size
        J = 1/(2*m) * np.sum(np.square(np.dot(X, self.theta) - y))
        return J

    def normalize(self, X):
        mu, sigma = np.mean(X, axis=0), np.std(X, axis=0)
        return (X - mu) / sigma

if __name__ == "__main__":

    data = np.loadtxt(os.path.join('data', 'profit.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    X = np.stack([np.ones(y.size), X], axis=1) # for theta[0]

    iterations = 1500
    alpha = 0.01

    model = LinearRegression(alpha, X.shape[1])
    theta, J_history  = model.partial_fit(X ,y, iterations)

    expected_theta = [-3.6303, 1.1664]
    for i in range(theta.size):
        assert(round(theta[i], 4) == expected_theta[i])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)

    # Plot data and the fitted line
    ax.plot(X[:, 1], y, 'ro', ms=10, mec='r')
    ax.plot(X[:, 1], model.predict(X), '-')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Training data', 'Linear regression']);

    # Plot the convergence graph
    ax = fig.add_subplot(122)
    ax.plot(np.arange(len(J_history)), J_history, lw=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    plt.show()
