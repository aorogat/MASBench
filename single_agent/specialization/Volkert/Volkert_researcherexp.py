
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('volkert.csv')

# Data Profiling
print("Data Shape:", data.shape)
print("Columns:", data.columns)
print("Data Types:\n", data.dtypes)
print("Unique Values:\n", data.nunique())

# Step 2: Data Cleaning
# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Convert numeric-like columns to numeric safely
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop or fix missing data
data = data.dropna(subset=['class'])  # Drop rows where target class is NaN
data = data.dropna(axis=1, how='any')  # Drop columns with any missing values

# Remove invalid 'class' entries if needed (this depends on the nature of the classes)
data = data[data['class'].notnull()]

# Step 3: Feature Engineering
# Keep only numeric columns for modeling
X = data[numeric_cols]
y = data['class']

# Impute missing numeric values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert back to a DataFrame
X = pd.DataFrame(X_imputed, columns=numeric_cols)

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Modeling
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a multiclass classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
