
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
df = pd.read_csv('EU-IT_cleaned.csv', delimiter=',')  # Adjust delimiter if necessary
print(df.head())

# Profile data
print(df.info())
print(df.isnull().sum())

# Separate features and target
X = df.drop('Position', axis=1)
y = df['Position']

# Step 2: Clean data - Handle missing values by dropping rows with any missing value
df_cleaned = df.dropna()
X_cleaned = df_cleaned.drop('Position', axis=1)
y_cleaned = df_cleaned['Position']

# Step 3: Engineer features

# Identify categorical and numerical features
categorical_features = X_cleaned.select_dtypes(include=['object']).columns
numerical_features = X_cleaned.select_dtypes(include=['int64', 'float64']).columns

# Impute missing numeric values with the mean
numeric_imputer = SimpleImputer(strategy='mean')
X_cleaned[numerical_features] = numeric_imputer.fit_transform(X_cleaned[numerical_features])

# Impute categorical values with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_cleaned[categorical_features] = categorical_imputer.fit_transform(X_cleaned[categorical_features])

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_cleaned[categorical_features])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Combine the encoded categorical features with the numerical features
X_final = pd.concat([X_cleaned[numerical_features].reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

# Scale numerical features
scaler = StandardScaler()
X_final[numerical_features] = scaler.fit_transform(X_final[numerical_features])

# Step 4: Model - Train a multiclass classifier
X_train, X_test, y_train, y_test = train_test_split(X_final, y_cleaned, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate - Predictions and performance metrics
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
print(classification_report(y_test, y_pred, zero_division=0))

# Display first 10 predictions vs actual labels
comparison_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(comparison_df.head(10))
