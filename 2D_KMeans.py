import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=100)
kmeans.fit(X_scaled)

# Obtener las etiquetas de los clústeres y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualizar los clusters en un subconjunto de características
plt.figure(figsize=(10, 6))

# Escoger dos características para visualizar
x_index = 0
y_index = 1

# Scatter plot de los datos coloreados por clúster
plt.scatter(X_scaled[:, x_index], X_scaled[:, y_index], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, x_index], centroids[:, y_index], marker='x', s=200, c='red')

# Etiquetas y título
plt.xlabel(iris.feature_names[x_index] + ' (scaled)')
plt.ylabel(iris.feature_names[y_index] + ' (scaled)')
plt.title('K-Means Clustering on Iris Dataset')

plt.colorbar(label='Cluster')
plt.show()
