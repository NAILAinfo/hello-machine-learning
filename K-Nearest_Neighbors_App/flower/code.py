from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Chargement du jeu de données Iris
iris = load_iris()
X = iris.data[:, 2:4]   # on garde 2 caractéristiques : longueur et largeur des pétales
y = iris.target

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prédictions
y_pred = knn.predict(X_test)

# Évaluation
print("✅ Précision du modèle :", accuracy_score(y_test, y_pred))
print("\n📋 Rapport de classification :\n", classification_report(y_test, y_pred))

# --- VISUALISATION ---

# Définition des limites du graphique
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# Création d'une grille de points
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Prédiction pour chaque point de la grille
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracé des zones de décision
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)

# Tracé des points d'entraînement et de test
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', marker='o', label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', marker='*', s=150, label="Test")

plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.title('Frontières de décision du KNN (k=3)')
plt.legend()
plt.show()
