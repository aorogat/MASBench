
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
file_path = 'Wifi.csv'
data = pd.read_csv(file_path)

# Step 1: Profile the Data
print("Data Info:")
print(data.info())
print("\nDistinct Values:")
print(data.nunique())
print("\nMissing Entries:")
print(data.isnull().sum())

# Step 2: Clean Data
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Step 3: Feature Engineering
# Convert 'TechCenter' to numeric labels
data['techcenter'] = data['techcenter'].map({'Yes': 1, 'No': 0})

# Impute missing values for the target column
imputer = SimpleImputer(strategy='most_frequent')
data['techcenter'] = imputer.fit_transform(data[['techcenter']])

# Separate features and target variable
X = data.drop('techcenter', axis=1)
y = data['techcenter']

# Categorical features processing
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=[np.number]).columns

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Step 4: Model
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train a binary classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)

# Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print Results
print("\nEvaluation Metrics:")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Print first 10 predicted and actual values
print("\nFirst 10 Predicted vs Actual values:")
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f'Predicted: {pred}, Actual: {actual}')
