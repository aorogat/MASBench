
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
data = pd.read_csv('Wifi.csv')

# Step 1: Profile the data
print("Data Types:\n", data.dtypes)
print("\nDistinct Values:\n", data.nunique())
print("\nMissing Entries:\n", data.isnull().sum())

# Step 2: Clean data - standardize column names
data.columns = [col.lower().replace(" ", "_") for col in data.columns]

# Step 3: Feature engineering
# Convert 'TechCenter' to numeric labels
data['techcenter'] = data['techcenter'].map({'Yes': 1, 'No': 0})

# Impute missing values for the target column
imputer = SimpleImputer(strategy='most_frequent')
data['techcenter'] = imputer.fit_transform(data[['techcenter']])

# Separate features and target variable
X = data.drop('techcenter', axis=1)
y = data['techcenter']

# One Hot Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Step 4: Split data into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a binary classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation results
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the first 10 predicted and actual values
pred_actual_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print("\nFirst 10 Predicted and Actual Values:\n", pred_actual_df.head(10))
