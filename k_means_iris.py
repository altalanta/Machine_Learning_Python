import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # True labels (not used for clustering)

# Create a K-Means clustering model with k=3 (since there are 3 species of Iris)
kmeans = KMeans(n_clusters=3, n_init=10)  # Set n_init explicitly to suppress the warning

# Fit the K-Means model to the data
kmeans.fit(X)

# Get the cluster labels assigned to each data point
labels = kmeans.labels_

# Add cluster labels to the original dataset
iris['cluster_labels'] = labels

# Visualize the clusters (for two features only, you can choose different pairs)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering of Iris Dataset')
plt.show()
