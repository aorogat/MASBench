
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the dataset
data = pd.read_csv('Utility.csv')

# 2. Clean column names
data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()

# 3. Drop rows with missing target
data = data.dropna(subset=['csri'])

# 4. Separate target and features
y = data['csri']
X = data.drop(columns=['csri'])

# 5. Identify numeric vs categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 6. Create transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 7. Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 8. Create a pipeline that includes the preprocessor and regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# 9. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Train the model
model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Evaluate the model and print MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# 13. Print first 10 predicted vs actual values
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(results.head(10))
