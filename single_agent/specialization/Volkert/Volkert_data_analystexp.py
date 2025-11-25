
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('volkert.csv')

# Data Profiling
print("Column Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nNumber of Unique Values:")
print(df.nunique())

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Data Cleaning
# Convert potential numeric-like columns to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing data
df.dropna(subset=['class'], inplace=True)  # Remove rows with missing 'class'
df.dropna(axis=1, how='any', inplace=True)  # Drop columns with any missing data
print("\nData shape after cleaning:", df.shape)

# Feature Engineering
# Keep only numeric columns
df_numeric = df[numeric_cols]
# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

# Define the target and features
X = df_imputed
y = df['class']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrics
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
