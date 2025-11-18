
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('Yelp_Merged.csv')

# 1) Data Profiling
print("Column Names:", data.columns.tolist())
print("\nData Types:\n", data.dtypes)
print("\nNumber of Unique Values:\n", data.nunique())
print("\nNumeric Columns:\n", data.select_dtypes(include=['int', 'float']).columns.tolist())
print("\nCategorical Columns:\n", data.select_dtypes(include=['object']).columns.tolist())

# 2) Data Cleaning
# Drop non-predictive identifiers and timestamp/date fields
data = data.drop(columns=['business_id', 'user_id', 'review_date'])

# Convert numeric-like columns to numeric safely
data = data.apply(pd.to_numeric, errors='ignore')

# Drop or fix columns that contain missing data
data = data.dropna(axis=1, how='any')

# Remove rows where the target 'stars' is missing or invalid
data = data[data['stars'].notnull() & (data['stars'].isin([1, 2, 3, 4, 5]))]

# 3) Feature Engineering
# Keep only numeric columns for modeling
numeric_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
X = data[numeric_features]
y = data['stars']

# Impute missing values and scale features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 4) Modeling
# Split the data into 80/20 train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the multiclass classifier
pipeline.fit(X_train, y_train)

# 5) Evaluation
y_pred = pipeline.predict(X_test)

# Compute Accuracy, Precision, Recall, and F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Print full classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
