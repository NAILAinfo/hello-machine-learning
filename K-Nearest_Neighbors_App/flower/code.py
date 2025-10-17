# Étape 1 : Importation des bibliothèques
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Étape 2 : Chargement du dataset
iris = load_iris()
X = iris.data[:, 2:4]   # 👉 on garde seulement 2 caractéristiques : longueur et largeur des pétales
y = iris.target

# Étape 3 : Division en données d’entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Création du modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Étape 5 : Entraînement du modèle
knn.fit(X_train, y_train)

# Étape 6 : Prédiction sur les données de test
y_pred = knn.predict(X_test)

# Étape 7 : Évaluation du modèle
print("✅ Précision du modèle :", accuracy_score(y_test, y_pred))
print("\n📋 Rapport de classification :\n", classification_report(y_test, y_pred))

# Étape 8 : Visualisation 2D
# Création d’une grille de points pour visualiser les zones de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Prédiction pour chaque point de la grille
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracé du fond coloré selon les classes prédites
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)

# Tracé des points réels du dataset
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)

# Légende et titres
plt.xlabel('Longueur du pétale (cm)')
plt.ylabel('Largeur du pétale (cm)')
plt.title('Classification KNN (k=3) sur 2D - Dataset Iris')
# Création d'une légende manuelle
colors = ['purple', 'green', 'orange']  # une couleur par espèce
for i, species in enumerate(iris.target_names):
    plt.scatter([], [], color=colors[i], label=species)

plt.legend(title="Espèces")

plt.show()
