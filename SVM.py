import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X = np.array([[1, 1], [2, 2], [2, 3], [3, 3],
 [6, 6], [7, 7], [7, 8], [8, 8]])
y = [0, 0, 0, 0, 1, 1, 1, 1]
model = SVC(kernel='linear', C=1000)
model.fit(X, y)
print("Support Vectors:")
print(model.support_vectors_)
x = np.linspace(0, 9, 100)
w = model.coef_[0]
b = model.intercept_[0]
boundary = -(w[0] * x + b) / w[1]
margin = 1 / np.linalg.norm(w)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(x, boundary, label="Maximum Margin Hyperplane")
plt.plot(x, boundary + margin, '--', label="Margin")
plt.plot(x, boundary - margin, '--')
plt.scatter(model.support_vectors_[:, 0],
 model.support_vectors_[:, 1],
 s=100, facecolors='none', edgecolors='black',
 label="Support Vectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM - Maximum Margin Hyperplane")
plt.legend()
plt.grid()
plt.show()

'''1. Import the required libraries.
2. Create a dataset with two classes.
3. Create and train the Linear SVM model.
4. Find and display the support vectors.
5. Calculate the hyperplane and margin.
6. Plot and display the hyperplane, margins and support vectors'''