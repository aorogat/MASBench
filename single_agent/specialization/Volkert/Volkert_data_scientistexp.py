
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('volkert.csv')

# Data Profiling
print("Initial Data Profiling")
print("Column Names:", data.columns.tolist())
print("Data Types:\n", data.dtypes)
print("Number of Unique Values:\n", data.nunique())

# Detect numeric vs categorical
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numeric Columns:", numeric_columns)
print("Categorical Columns:", categorical_columns)

# Step 2: Data Cleaning
# Convert numeric-like columns to numeric safely
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop or fix columns that contain missing data
data = data.dropna(axis=1, how='any')  # Drop columns with any missing values

# Remove rows where the target 'class' is missing or invalid
data = data[data['class'].notna()]

# Step 3: Feature Engineering
# Keep only numeric columns for modeling
X = data[numeric_columns]
y = data['class']

# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 4: Modeling
# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a multiclass classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = classifier.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
