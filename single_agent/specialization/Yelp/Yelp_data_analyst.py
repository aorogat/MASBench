
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')

# Handle missing data
df = df.dropna()

# Drop non-numeric columns
df = df.drop(columns=['business_id', 'user_id', 'review_date'])

# Separate features and target variable
X = df.drop(columns=['stars'])
y = df['stars']

# Apply encoding for categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print first 10 predicted and actual stars values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 10 Actual and Predicted Stars Values:\n", results.head(10))
