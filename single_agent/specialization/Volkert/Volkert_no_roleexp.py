
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1) Load the dataset
df = pd.read_csv('volkert.csv')

# 1) Data Profiling
print("Data Profiling:")
print(df.info())  # Inspect column information including data types
print("Number of unique values in each column:\n", df.nunique()) # Number of unique values
print("DataFrame shape:", df.shape)  # Shape of the DataFrame
print("Columns:", df.columns.tolist())  # List of column names
print("First few rows of the dataset:\n", df.head())  # Preview the data

# Determine numeric vs categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# 2) Data Cleaning
# Convert all numeric-like columns to numeric safely (with errors='coerce' to handle conversion issues)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop or fix columns that contain missing data
df.dropna(axis=1, how='any', inplace=True)  # Dropping columns with any missing value
print("Remaining columns after dropping NA columns:", df.columns.tolist())

# Remove rows where the target 'class' is missing or invalid
df = df[df['class'].notnull()]

# 3) Feature Engineering
X = df[numeric_cols].copy()  # Features containing only numeric columns
y = df['class']  # Target variable

# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# Optionally scale features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 4) Modeling
# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a RandomForestClassifier
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# 5) Evaluation
y_pred = pipeline.predict(X_test)

# Compute metrics
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
print("Classification Report:\n", classification_report(y_test, y_pred))
