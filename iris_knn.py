# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Load a dataset (Iris dataset in this example)
data = load_iris()
X = data.data  # Features
y = data.target  # Target variable (labels)

# Create a KNN classifier with a specified number of neighbors (k)
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Perform k-fold cross-validation (e.g., with k=5 folds)
num_folds = 5  # You can change this value as needed
scores = cross_val_score(knn_classifier, X, y, cv=num_folds)

# Print the cross-validation scores
for fold, score in enumerate(scores, start=1):
    print(f"Fold {fold} Accuracy: {score:.2f}")

# Calculate and print the average accuracy
average_accuracy = sum(scores) / num_folds
print(f"Average Accuracy: {average_accuracy:.2f}")
