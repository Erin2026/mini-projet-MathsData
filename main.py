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
print(X)
y = (digits.target == 1).astype(int)  # Classification binaire : "1" vs "non-1"
print(y)
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajouter le biais
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialiser les poids
theta_init = np.zeros(X_train_bias.shape[1])

# Entraîner le modèle
gd = GradientDescent(gradient=cost_gradient, learning_rate=0.1, max_iterations=10000, epsilon=1e-5)
theta_opt = gd.descent(theta_init, data=(X_train_bias, y_train))

# Prédictions
y_pred = predict(theta_opt, X_test_bias)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle (Descente de Gradient) : {accuracy * 100:.2f}%")

# Comparaison avec scikit-learn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
sklearn_accuracy = logreg.score(X_test, y_test)
print(f"Précision du modèle (scikit-learn) : {sklearn_accuracy * 100:.2f}%")
