
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('volkert.csv')

# Drop non-numeric columns and handle missing data
df = df.select_dtypes(include=['float64', 'int64'])
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the accuracy, precision, recall, and F1 score
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Print the first 10 predicted and actual values
print("First 10 Predicted vs Actual:")
for i in range(10):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}')
