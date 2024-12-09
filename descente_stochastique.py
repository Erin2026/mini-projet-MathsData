import numpy as np

class GradientDescent:
    def __init__(self, gradient, learning_rate=0.01, max_iterations=1000, epsilon=1e-6, batch_size=1):
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size

    def descent(self, initial_point, data=None):
        current_point = initial_point
        for epoch in range(self.max_iterations):
            if data is not None:
                X, y = data
                dataset = np.c_[X, y]
                np.random.shuffle(dataset)
                X_shuffled = dataset[:, :-1]
                y_shuffled = dataset[:, -1]

                for i in range(0, len(X_shuffled), self.batch_size):
                    mini_batch_X = X_shuffled[i:i + self.batch_size]
                    mini_batch_y = y_shuffled[i:i + self.batch_size]
                    gradient_value = self.gradient(current_point, mini_batch_X, mini_batch_y)
                    if np.linalg.norm(gradient_value) < self.epsilon:
                        return current_point
                    current_point -= self.learning_rate * gradient_value
            else:
                gradient_value = self.gradient(current_point)
                if np.linalg.norm(gradient_value) < self.epsilon:
                    return current_point
                current_point -= self.learning_rate * gradient_value
        return current_point
