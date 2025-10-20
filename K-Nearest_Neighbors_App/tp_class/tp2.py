import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error

x_train = np.array([
    [1.0, 0.25],
    [0.4, 0.10],
    [0.5, 0.50],
    [1.0, 1.0]
])

y_train = np.array([0, 0, 1, 1])

p = np.array([[0.5, 0.15]])  # Doit être 2D pour sklearn


k = 3  
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

prediction = knn.predict(p)
print(f"Classe prédite pour le point {p} : {prediction[0]}")

# ---- Visualisation ----
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', label='Train data')
plt.scatter(p[0, 0], p[0, 1], color='green', marker='X', s=150, label='Point à prédire')
plt.title("KNN Classification")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
