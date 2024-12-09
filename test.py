import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from descente_stochastique_gpt import GradientDescent


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_gradient(theta, X, y):
    m = len(y)
    predictions = sigmoid(X @ theta)
    errors = predictions - y
    return (1 / m) * (X.T @ errors)


def predict(theta, X):
    return (sigmoid(X @ theta) >= 0.5).astype(int)


# Charger le dataset
digits = load_digits()
X = digits.data / 16.0
y = (digits.target == 1).astype(int)  # Classification binaire : "1" vs "non-1"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajouter le biais
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]


# Initialiser les poids
theta_init = np.zeros(X_train_bias.shape[1])
print(X_train_bias.shape[1])