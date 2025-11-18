
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('volkert.csv')

# Handle missing data
data = data.dropna()

# Drop non-numeric columns
data_numeric = data.select_dtypes(include=[np.number])

# Assume 'class' is the target column
X = data_numeric.drop('class', axis=1)
y = data['class']

# Encode the target variable if it's categorical
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print first 10 predicted and actual values
print("\nFirst 10 predicted and actual values:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f'Actual: {actual}, Predicted: {predicted}')
