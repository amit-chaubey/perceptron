import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset
X = np.array([[2, 3], [4, 1], [1, 3], [3, 4], [5, 2]])
y = np.array([1, 1, -1, -1, 1])  # Labels: 1 for positive, -1 for negative

# Initial weights and bias
W = np.array([0.1, -0.2])
b = 0.1

# Perceptron update rule
def perceptron_update(W, b, X, y):
    for X_i, y_i in zip(X, y):
        if y_i * (np.dot(W, X_i) + b) <= 0:
            W = W + y_i * X_i
            b = b + y_i
    return W, b

# Function to plot the decision boundary
def plot_decision_boundary(W, b, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=100, edgecolors='k')
    ax = plt.gca()
    
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * W[0] + b) / W[1]
    plt.plot(x_vals, y_vals)

# Plot initial decision boundary
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plot_decision_boundary(W, b, X, y)
plt.title('Initial Decision Boundary')

# Perform one update for a misclassified instance
W_updated, b_updated = perceptron_update(W, b, X, y)

# Plot updated decision boundary
plt.subplot(1, 2, 2)
plot_decision_boundary(W_updated, b_updated, X, y)
plt.title('Updated Decision Boundary')

plt.tight_layout()
plt.show()
