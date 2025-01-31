import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Projet.descente_stochastique import GradientDescent


def softmax(z):
    """
    Calcule la fonction softmax pour une matrice ou un vecteur d'entrées.

    La fonction softmax convertit un vecteur de scores en un vecteur de
    probabilités, où chaque élément est dans l'intervalle [0, 1] et la somme
    des éléments est égale à 1.

    Paramètres :
    - z : Un tableau ou une matrice contenant les scores (logits) pour chaque
    exemple.

    Retourne :
    - Un tableau de probabilités où chaque élément représente la probabilité de
    la classe associée.
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def E(theta, X, Y):
    """
    Calcule la fonction de coût pour la régression logistique multiclasses
    en utilisant la fonction softmax.

    Paramètres :
    - theta : Matrice des paramètres (coefficients) du modèle, de forme
    nb_classes*nb_parametres.
    - X : Matrice des entrées (données), de forme nb_images*nb_parametres.
    - Y : Matrice des sorties réelles (labels), de forme
    nb_images*nb_classes.

    Retourne :
    - float
      Le coût total, calculé comme la somme des log-vraisemblances négatives.
    """
    scores = np.matmul(X, theta.T)
    probabilities = softmax(scores)
    nb_images = X.shape[0]
    nb_classes = Y.shape[1]

    cost = 0
    for i in range(nb_images):
        for c in range(nb_classes):
            cost += -Y[i, c] * np.log(probabilities[i, c])

    return cost


def gradE(theta, data):
    """
    Calcule le gradient de la fonction de coût par rapport aux paramètres
    theta pour la régression logistique.

    Paramètres :
    - theta : Matrice des paramètres du modèle.
    - data : Données d'entraînement avec les entrées et les labels en une seule
    matrice, de forme nb_images*(nb_features + nb_classes).

    Retourne :
    - Le gradient de la fonction de coût, de même forme que theta.
    """
    X, Y = data[:, :-10], data[:, -10:]
    scores = np.matmul(X, theta.T)
    probabilities = softmax(scores)
    return np.matmul((probabilities - Y).T, X)


# Chargement des données
digits = load_digits()
X = digits.data  # 64 données d'entrée par image
Y = digits.target.reshape(-1, 1)  # Labels (0-9)

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ajout du biais
X = np.c_[np.ones(X.shape[0]), X]

# Encodage des labels en one-hot
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(Y)

# Division des données en train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2)

# --------------------
# Descente de gradient personnalisée
# --------------------
# Concaténation de X_train et Y_train pour la descente de gradient
train_data = np.hstack((X_train, Y_train))

# Initialisation des paramètres
nb_classes = 10
theta_init = np.zeros((nb_classes, X_train.shape[1]))

# Initialisation de la descente
delta = 0.01
gd = GradientDescent(gradE, learning_rate=delta, max_iterations=1000,
                     batch_size=len(X_train))

# Apprentissage
theta_final = gd.descent(theta_init, data=train_data)
print("Erreur finale (train) (implémentation personnalisée) :",
      E(theta_final, X_train, Y_train))

# Prédiction sur le jeu de test
probas_test = softmax(np.matmul(X_test, theta_final.T))
y_pred_custom = np.argmax(probas_test, axis=1)
y_test_labels = np.argmax(Y_test, axis=1)

# Calcul du taux de bonne classification
accuracy_custom = np.mean(y_pred_custom == y_test_labels)

# Calcul de la matrice de confusion (implémentation personnalisée)
conf_matrix_custom = confusion_matrix(y_test_labels, y_pred_custom)
disp_custom = ConfusionMatrixDisplay(conf_matrix_custom)
disp_custom.plot(cmap='Blues')
plt.title("Matrice de confusion - Implémentation personnalisée")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.show()

# --------------------
# Descente de gradient avec sklearn
# --------------------
# Initialisation de LogisticRegression
model = LogisticRegression(max_iter=1000)

# Conversion des labels en format 1D pour sklearn
Y_train_sklearn = np.argmax(Y_train, axis=1)
Y_test_sklearn = np.argmax(Y_test, axis=1)

# Entraînement du modèle
model.fit(X_train, Y_train_sklearn)

# Prédictions sur le jeu de test
y_pred_sklearn = model.predict(X_test)

# Calcul du taux de bonne classification
accuracy_sklearn = np.mean(y_pred_sklearn == Y_test_sklearn)

# Calcul de la matrice de confusion (avec sklearn)
conf_matrix_sklearn = confusion_matrix(Y_test_sklearn, y_pred_sklearn)
disp_sklearn = ConfusionMatrixDisplay(conf_matrix_sklearn)
disp_sklearn.plot(cmap='Blues')
plt.title("Matrice de confusion - Sklearn")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.show()

# --------------------
# Affichage comparatif des résultats
# --------------------
print("\nComparaison des précisions sur le test :")
print(f"Implémentation personnalisée : {accuracy_custom * 100:.2f}%")
print(f"Sklearn : {accuracy_sklearn * 100:.2f}%")

# --------------------
# Quelques exemples
# --------------------
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i, 1:].reshape(8, 8), cmap='gray')
    ax.set_title(f'Vrai chiffre: {y_test_labels[i]}\n'
                 f'Notre descente: {y_pred_custom[i]}\n'
                 f'Descente Sklearn: {y_pred_sklearn[i]}')
    ax.axis('off')
plt.show()
