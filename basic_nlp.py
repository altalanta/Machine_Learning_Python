import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Sample text data
corpus = [
    "I love NLP!",
    "NLP is fascinating.",
    "Text classification is important in NLP.",
    "Machine learning algorithms are used in NLP.",
    "Python is a popular language for NLP tasks."
]

# Corresponding labels (0 for non-NLP, 1 for NLP)
labels = [1, 0, 1, 1, 0]

# Create a CountVectorizer to convert text data to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

svm_classifier = SVC(kernel='rbf')  # Third-degree polynomial kernel

# Perform cross-validation and get accuracy scores
accuracy_scores = cross_val_score(svm_classifier, X, labels, cv=2)  # 5-fold cross-validation

# Print the accuracy scores for each fold
print("Accuracy scores for each fold:", accuracy_scores)

# Calculate the mean accuracy
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)
