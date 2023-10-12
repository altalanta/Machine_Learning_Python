import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target variable (labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a simple feedforward neural network
model = keras.Sequential([
    layers.Input(shape=(4,)),  # Input layer with 4 neurons (Iris dataset has 4 features)
    layers.Dense(units=128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    layers.Dense(units=3, activation='softmax')  # Output layer with 3 neurons (for 3-class classification) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',  # Optimizer (Adam is a popular choice)
              loss='sparse_categorical_crossentropy',  # Loss function for classification
              metrics=['accuracy'])  # Evaluation metric

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
