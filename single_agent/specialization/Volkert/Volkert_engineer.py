
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('volkert.csv')

# Drop non-numeric columns and handle missing data
data = data.select_dtypes(include=[np.number])  # Keep only numeric columns
data.dropna(inplace=True)  # Drop rows with missing values

# Separate features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Encode target variable if not already numeric
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print first 10 predicted and actual values
print("\nFirst 10 Predicted and Actual Values:")
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f'Predicted: {pred}, Actual: {actual}')
