
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load and Profile the dataset
data = pd.read_csv('volkert.csv')

# Inspect column names, data types, and number of unique values
print("Column Names:", data.columns)
print("Data Types:", data.dtypes)
print("Unique Values:", data.nunique())

# Detect numeric vs categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# Step 2: Data Cleaning
# Convert all numeric-like columns to numeric safely
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop or fix columns that contain missing data
data = data.dropna(axis=1, how='any')  # Drop columns with any missing data
data = data.dropna(subset=['class'])    # Remove rows where the target 'class' is missing or invalid

# Step 3: Feature Engineering
# Keep only numeric columns for modeling
X = data[numeric_cols]

# Impute missing numeric values with the mean
X.fillna(X.mean(), inplace=True)

# Optionally scale features (you can choose to scale if needed, here we skip it for simplicity)

# Prepare target variable
y = data['class']

# Step 4: Modeling
# Train an 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = classifier.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print the full classification report
print(classification_report(y_test, y_pred))
