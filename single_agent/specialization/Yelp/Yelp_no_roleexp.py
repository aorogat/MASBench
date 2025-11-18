
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('Yelp_Merged.csv')

# 1) Profile Data
print("Data Overview:")
print(data.info())
print("\nMissing Values Summary:")
print(data.isnull().sum())

# 2) Clean Data
# Dropping irrelevant columns
data_cleaned = data.drop(columns=['business_id', 'user_id', 'review_date'], errors='ignore')

# 3) Engineer features
# Selecting numeric columns
numeric_data = data_cleaned.select_dtypes(include=['float64', 'int64'])

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
numeric_imputed = imputer.fit_transform(numeric_data)

# Convert to DataFrame
numeric_data = pd.DataFrame(numeric_imputed, columns=numeric_data.columns)

# Optionally scale features
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_data)

# Final dataset with features and target
X = pd.DataFrame(numeric_scaled, columns=numeric_data.columns)
y = data_cleaned['stars']

# 4) Model
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print results
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
