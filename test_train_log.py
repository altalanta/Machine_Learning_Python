import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Specify the path to the downloaded Excel file
dataset_path = "/Users/yourusername/Downloads/logistic_dataset.xlsx"

# Load the dataset from the Excel file
df = pd.read_excel(dataset_path)

# Split the dataset into features (X) and target (y)
X = df[['Feature']]
y = df['Target']

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance (you can use different metrics)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
