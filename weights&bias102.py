import numpy as np
import matplotlib.pyplot as plt

# Generate a more sophisticated dataset
np.random.seed(42)  # For reproducibility
X_positive = np.random.randn(20, 2) + np.array([2, 2])  # Positive class centered around (2,2)
X_negative = np.random.randn(20, 2) + np.array([-2, -2])  # Negative class centered around (-2,-2)
X = np.vstack((X_positive, X_negative))
y = np.array([1] * 20 + [-1] * 20)  # Labels: 1 for positive, -1 for negative

# Initialize weights and bias
W = np.random.randn(2)
b = np.random.randn()

# Learning rate
alpha = 0.1

# Perceptron update rule with a learning rate
def perceptron_update(W, b, X, y, alpha):
    for X_i, y_i in zip(X, y):
        if y_i * (np.dot(W, X_i) + b) <= 0:
            W = W + alpha * y_i * X_i  # Update the weight vector
            b = b + alpha * y_i  # Update the bias
    return W, b

# Function to plot the decision boundary
def plot_decision_boundary(W, b, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=100, edgecolors='k')
    ax = plt.gca()
    
    x_vals = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    y_vals = -(x_vals * W[0] + b) / W[1]
    plt.plot(x_vals, y_vals, '--k')
    
    plt.xlim(min(X[:,0]) - 1, max(X[:,0]) + 1)
    plt.ylim(min(X[:,1]) - 1, max(X[:,1]) + 1)
    plt.title(title)

# Plot initial decision boundary
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plot_decision_boundary(W, b, X, y, 'Initial Decision Boundary')

# Perform one update for a misclassified instance
W_updated, b_updated = perceptron_update(W, b, X, y, alpha)

# Plot updated decision boundary
plt.subplot(1, 2, 2)
plot_decision_boundary(W_updated, b_updated, X, y, 'Updated Decision Boundary')

plt.tight_layout()
plt.show()
