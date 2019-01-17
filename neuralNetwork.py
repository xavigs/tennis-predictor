#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - x ** 2

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivative
            
        # Init weights
        self.weights = []
        self.deltas = []
        
        # Set random values
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
            
        r = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
        
    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
 
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
                
            self.deltas.append(deltas)
            deltas.reverse()
 
            # Backpropagation
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
 
            if k % 10000 == 0: print('epochs:', k)
        
    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
 
    def print_weights(self):
        print("Llistat de pesos de connexions")
        for i in range(len(self.weights)):
            print(self.weights[i])
 
    def get_deltas(self):
        return self.deltas

# Test app (funció AND)
nn = NeuralNetwork([2, 2, 1], activation = 'tanh')

X = np.array([[3, 2],
              [1, 7],
              [8, 4],
              [5, 5],
              [0, 6],
              [7, 7],
              [1, 1],
              [9, 3],
              [4, 9],
              [2, 0],
              [0, 4],
              [5, 2],
              [3, 5],
              [8, 8],
              [9, 9],
              [1, 0],
              [0, 0],
              [2, 4]])
 
y = np.array([[0.05],
              [0.08],
              [0.12],
              [0.10],
              [0.06],
              [0.14],
              [0.02],
              [0.12],
              [0.13],
              [0.02],
              [0.04],
              [0.07],
              [0.08],
              [0.16],
              [0.18],
              [0.01],
              [0.00],
              [0.06]])

Z = np.array([[4, 4],
              [1, 3],
              [0, 2],
              [9, 6]])

sol = np.array([[0.08],
               [0.04],
               [0.02],
               [0.15]])

nn.fit(X, y, learning_rate = 0.03, epochs = 200000)

index = 0

for e in Z:
    print("Z:", e, "Sol:", sol[index], "Network:", nn.predict(e))
    index = index + 1

