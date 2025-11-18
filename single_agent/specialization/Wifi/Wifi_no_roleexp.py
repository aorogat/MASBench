
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
df = pd.read_csv('Wifi.csv')

# Step 1: Profile the data
print("Data Profiling:")
print(df.info())
print(df.describe())
print("Missing Values:\n", df.isnull().sum())
print("Distinct Values in 'TechCenter':\n", df['TechCenter'].value_counts())

# Step 2: Clean data - standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 3: Feature engineering
# Convert 'TechCenter' to numeric labels
df['techcenter'] = df['techcenter'].map({'Yes': 1, 'No': 0})

# Impute missing values for the target column (if any)
imputer = SimpleImputer(strategy='most_frequent')
df['techcenter'] = imputer.fit_transform(df[['techcenter']])

# Split the features and target
X = df.drop(columns=['techcenter'])
y = df['techcenter']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

# Create a ColumnTransformer with OneHotEncoder for categorical columns and StandardScaler for numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

# Step 4: Model - split data into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and a classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation results
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the first 10 predicted and actual values
print("\nFirst 10 Predicted and Actual 'TechCenter' Values:")
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(results.head(10))
