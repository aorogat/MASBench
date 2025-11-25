
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the dataset
df = pd.read_csv('Yelp_Merged.csv')

# 2. Data Profiling
print("Column Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nNumber of Unique Values:\n", df.nunique())

# 3. Inspect numeric vs categorical
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nNumeric Columns:\n", numeric_cols)
print("\nCategorical Columns:\n", categorical_cols)

# 4. Data Cleaning
# Drop non-predictive identifiers
df_cleaned = df.drop(columns=['business_id', 'user_id', 'review_date'], errors='ignore')

# Convert numeric-like columns to numeric safely
for col in df_cleaned.columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')

# Drop or fix columns with missing data
df_cleaned = df_cleaned.dropna(axis=1, how='any')

# Remove rows where the target 'stars' is missing or invalid
df_cleaned = df_cleaned[df_cleaned['stars'].notnull()]

# 5. Feature Engineering
# Keep only numeric columns for modeling
feature_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
X = df_cleaned[feature_cols]
y = df_cleaned['stars']

# Impute missing values in numeric columns
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 6. Modeling
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluation
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))