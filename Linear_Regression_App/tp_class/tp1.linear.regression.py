import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 1. Generate sample data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7])
# 2. Create and train the model
model = LinearRegression()
model.fit(X, y)
# 3. Display model parameters
print("Coefficient (a):", model.coef_[0])
print("Intercept (b):", model.intercept_)
# 4. Make predictions
X_new = np.array([[7], [8], [9]])
y_pred = model.predict(X_new)
print("\nPredictions for X = 7, 8, 9 :", y_pred)
# 5. Visualization
plt.scatter(X, y, color='blue', label="Training data")
plt.plot(X, model.predict(X), color='red', label="Regression line")
plt.scatter(X_new, y_pred, color='green', label="Predictions")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
