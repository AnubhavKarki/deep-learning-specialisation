import numpy as np

# Reproducible random data
np.random.seed(0)

# Configuration
n_features = 3     # number of input features
m_examples = 120   # number of training examples
n_hidden = 4       # number of neurons in hidden layer

# ----- Input -----
X = np.random.randn(n_features, m_examples)   # shape: (3, 120)

# ----- Hidden layer parameters -----
W1 = np.random.randn(n_hidden, n_features)    # shape: (4, 3)
b1 = np.random.randn(n_hidden, 1)             # shape: (4, 1)

# ----- Forward pass: hidden layer -----
Z1 = W1 @ X + b1                              # shape: (4, 120)
A1 = 1 / (1 + np.exp(-Z1))                    # sigmoid activation, shape: (4, 120)

# ----- Output layer parameters -----
W2 = np.random.randn(1, n_hidden)             # shape: (1, 4)
b2 = np.random.randn(1, 1)                    # shape: (1, 1)

# ----- Forward pass: output layer -----
Z2 = W2 @ A1 + b2                             # shape: (1, 120)
A2 = 1 / (1 + np.exp(-Z2))                    # final predictions, shape: (1, 120)

# ----- Display shapes -----
print("Matrix shapes:")
print("  X   :", X.shape)
print("  W1  :", W1.shape)
print("  b1  :", b1.shape)
print("  Z1  :", Z1.shape)
print("  A1  :", A1.shape)
print("  W2  :", W2.shape)
print("  b2  :", b2.shape)
print("  Z2  :", Z2.shape)
print("  A2  :", A2.shape)

# ----- Show first 5 columns (examples) for inspection -----
print("\nFirst 5 columns of X (inputs):")
print(X[:, :5])

print("\nFirst 5 columns of A1 (hidden activations):")
print(A1[:, :5])

print("\nFirst 5 columns of A2 (output predictions):")
print(A2[:, :5])
