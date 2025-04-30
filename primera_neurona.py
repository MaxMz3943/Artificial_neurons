import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(1)
pesos = 2 * np.random.random((2, 1)) - 1
bias = np.random.rand()
tasa_aprendizaje = 0.1
for i in range(10000):
  
    entrada = np.dot(X, pesos) + bias
    salida = sigmoid(entrada)
    error = y - salida
    
    ajustes = error * sigmoid_derivada(salida)
    pesos += np.dot(X.T, ajustes) * tasa_aprendizaje
    bias += np.sum(ajustes) * tasa_aprendizaje
    
print("Salida después del entrenamiento:")
print(salida)
