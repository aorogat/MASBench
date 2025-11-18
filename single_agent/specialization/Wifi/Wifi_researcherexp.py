
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
data = pd.read_csv('Wifi.csv')

# Step 2: Profile the data
print("Data types:\n", data.dtypes)
print("\nSummary of distinct values:\n", data.nunique())
print("\nMissing entries:\n", data.isnull().sum())

# Step 3: Clean data - standardize column names
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Step 4: Feature engineering
# Convert 'TechCenter' to numeric labels
data['techcenter'] = data['techcenter'].map({'Yes': 1, 'No': 0})

# Impute missing values for the target column if any
data['techcenter'] = data['techcenter'].fillna(data['techcenter'].mode()[0])

# Define feature columns and the target
X = data.drop('techcenter', axis=1)
y = data['techcenter']

# Prepare numeric and categorical feature lists (example columns)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Creating the preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the complete pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42))])

# Step 5: Split the data into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = pipeline.predict(X_test)

# Fetching the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print the first 10 predicted and actual values
print("\nFirst 10 predicted and actual values:")
for i in range(10):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
