
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
file_path = 'Yelp_Merged.csv'
data = pd.read_csv(file_path)

# Data Profiling
print("Column Names and Data Types:")
print(data.dtypes)
print("\nNumber of Unique Values:")
print(data.nunique())
print("\nData Sample:")
print(data.head())

# Step 2: Data Cleaning
# Drop non-predictive identifiers
data.drop(columns=['business_id', 'user_id', 'review_date'], inplace=True)

# Convert numeric-like columns to numeric safely
for col in data.select_dtypes(include='object').columns:
    data[col] = pd.to_numeric(data[col], errors='ignore')

# Drop or fix columns that contain missing data
data.dropna(axis=1, how='any', inplace=True)
# Also, drop rows with missing target values
data = data[data['stars'].notnull() & (data['stars'].isin([1, 2, 3, 4, 5]))]

# Step 3: Feature Engineering
# Assuming all remaining columns are numeric
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
X = data[numeric_features]
y = data['stars']

# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Optionally scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = classifier.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Print full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
