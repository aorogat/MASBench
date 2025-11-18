
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('EU-IT_cleaned.csv')

# Profile data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Clean data: Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['Position'])
y = df['Position']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Impute missing numeric values with mean and categorical values with most frequent
num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

# Scale numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols)

# Combine processed features
X_processed = pd.concat([X_scaled, X_encoded], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train multiclass classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

# Print classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Display first 10 predictions vs actual labels
predictions_vs_actual = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(predictions_vs_actual.head(10))
