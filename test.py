import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from descente_stochastique_gpt import GradientDescent


def neurone(thetai,x):                  #fonction de prédiction d'un neurone
    y = 0
    for i in range(len(x)):
        y += np.dot(thetai[i],x[i])
    return y

def totalneurone(theta,x):             #division par la somme des exp de chaque neurone
    div = np.array([0]*len(theta))
    for i in range(len(theta)):
        div[i] = neurone(theta[i],x)
    return np.sum(div)

def softmax(theta,z):
    result = np.array([0]*len(theta))
    totalneurone = totalneurone(theta,z)
    for i in range(len(theta)):
        y = neurone(theta[i],z)
        result[i] = np.exp(y)/totalneurone
    return z

def errorSoftmax(theta,x,y):
    error = np.array([0]*len(theta[0]))
    for i in range(len(theta)):
        error[i] = -y[i]*np.log(softmax(theta,x))
    return error

def gradientErrorSoftmax(theta,x,y):
    gradError = np.array([0]*len(theta[0]))
    for i in range(len(theta)):
        gradError[i] = x*(softmax(theta,x)-y[i])
    return gradError


digits = load_digits()
print(digits.data)
scaler = StandardScaler()
X = scaler.fit_transform(digits.data)
print(X.shape)

y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

theta_init = np.array([np.zeros(X_train_bias.shape[1])]*len(digits.target_names))
print(theta_init.shape)