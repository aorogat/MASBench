
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Yelp_Merged.csv')

# 1) Data Profiling
print("Column Names:", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print("Number of Unique Values:\n", df.nunique())
print("Numeric Columns:\n", df.select_dtypes(include=[np.number]).columns.tolist())
print("Categorical Columns:\n", df.select_dtypes(include=['object']).columns.tolist())

# 2) Data Cleaning
# Dropping non-predictive identifiers and date/time fields
df.drop(columns=['business_id', 'user_id', 'review_date'], inplace=True)

# Convert numeric-like columns to numeric safely
df = df.apply(pd.to_numeric, errors='ignore')

# Dropping or fixing columns that contain missing data
# Example: Dropping columns if a very high percentage of data is missing
missing_threshold = 0.25
df = df.loc[:, df.isnull().mean() < missing_threshold]

# Remove rows where the target 'stars' is missing or invalid
df = df[df['stars'].notnull() & df['stars'].isin([1, 2, 3, 4, 5])]

# 3) Feature Engineering
# Selecting only numeric columns for modeling
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_columns].drop('stars', axis=1)
y = df['stars']

# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4) Modeling
# Train a multiclass classifier (RandomForestClassifier)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# 5) Evaluation
y_pred = classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print(classification_report(y_test, y_pred))
