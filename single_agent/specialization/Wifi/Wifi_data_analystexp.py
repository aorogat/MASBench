
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Wifi.csv')

# Profile the data
print("Dataset Info:")
print(df.info())
print("\nSummary of distinct values:")
print(df.nunique())
print("\nMissing values:")
print(df.isnull().sum())

# Clean data
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Feature engineering
# Convert 'TechCenter' to numeric labels
df['techcenter'] = df['techcenter'].map({'Yes': 1, 'No': 0})

# Impute missing values for the target column
df['techcenter'].fillna(df['techcenter'].mode()[0], inplace=True)

# Define the features and target variable
X = df.drop('techcenter', axis=1)
y = df['techcenter']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Create transformers for categorical and numerical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Bundle the preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())])

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Print the first 10 predicted and actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nFirst 10 predicted and actual values:")
print(results.head(10))
