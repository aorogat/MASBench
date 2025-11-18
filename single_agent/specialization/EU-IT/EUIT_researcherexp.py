
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('EU-IT_cleaned.csv', delimiter=',')  # Adjust the delimiter if necessary

# Profile data
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print("Categorical Features:", df.select_dtypes(include=['object']).columns.tolist())
print("Numerical Features:", df.select_dtypes(include=['float64', 'int64']).columns.tolist())

# Clean data
df.dropna(inplace=True)  # Drop rows with missing values

# Feature engineering
# Separate features and target
X = df.drop('Position', axis=1)
y = df['Position']

# Impute missing values for numerical features with mean
for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col].fillna(X[col].mean(), inplace=True)

# Impute missing values for categorical features with the most frequent value
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

# Combine features back
import numpy as np

X_final = np.concatenate([X_scaled, X_encoded], axis=1)

# Train a multiclass classifier
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Initialize and fit the classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

# Print Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Display first 10 predictions vs actual labels
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head(10))
