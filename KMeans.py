import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = np.array([[1, 35], [2, 40], [2, 45], [3, 50],
 [7, 75], [8, 80], [8, 85], [9, 90]])
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)
print("Cluster Labels:")
print(model.labels_)
print("Cluster Centers:")
print(model.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0],
 model.cluster_centers_[:, 1],
 marker='X', s=200)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("K-Means Clustering")
plt.show()

'''1. Import the required libraries.
2. Create a dataset containing Study Hours and Marks.
3. Create a K-Means model with 2 clusters.
4. Train the model and group the data into clusters.
5. Display cluster labels and cluster centres.
6. Plot and display the clusters.
'''