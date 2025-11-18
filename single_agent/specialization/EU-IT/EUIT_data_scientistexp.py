
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
file_path = 'EU-IT_cleaned.csv'
df = pd.read_csv(file_path, delimiter=',')  # Assuming comma as the delimiter; change if needed

# Profile data
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print("First 5 rows:\n", df.head())

# Step 2: Clean data
# Dropping rows with any missing values
df = df.dropna()

# Step 3: Feature engineering
# Separate features and target variable
X = df.drop(columns=['Position'])
y = df['Position']

# Handling numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Impute missing numeric values with mean
num_imputer = SimpleImputer(strategy='mean')
X[numerical_features] = num_imputer.fit_transform(X[numerical_features])

# Impute categorical values with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid dummy variable trap
encoded_categorical = encoder.fit_transform(X[categorical_features])

# Replace categorical columns with encoded columns
X = X.drop(columns=categorical_features)
X = np.hstack((X.values, encoded_categorical))

# Scale numerical features
scaler = StandardScaler()
X[:, :len(numerical_features)] = scaler.fit_transform(X[:, :len(numerical_features)])

# Step 4: Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate performance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Display the first 10 predictions vs actual labels
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 10 Predictions vs Actual Labels:\n", comparison_df.head(10))
