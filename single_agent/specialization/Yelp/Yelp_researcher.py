
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
data = pd.read_csv('Yelp_Merged.csv')

# Step 2: Handle missing data
data = data.dropna()

# Step 3: Drop non-numeric columns
data = data.drop(columns=['business_id', 'user_id', 'review_date'])

# Step 4: Separate features and target variable
X = data.drop('stars', axis=1)
y = data['stars']

# Step 5: Encode categorical features if any
X = pd.get_dummies(X, drop_first=True)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Step 10: Print results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(1, 6)]))

# Print first 10 predicted and actual stars values
print('\nFirst 10 Predicted and Actual stars values:')
for i in range(10):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}')
