import pandas as pd

def generate_model_code(file_path: str, target: str = None):
    # If target is not provided, try to guess (usually the last column) or just leave a placeholder
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    
    if target is None:
        target = columns[-1] # Default to last column
    
    feature_cols = [c for c in columns if c != target]
    
    code = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Load Data
df = pd.read_csv("data.csv") # Replace with your file path

# Target and Features
target = "{target}"
features = {feature_cols}

X = df[features]
y = df[target]

# Detect numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define Model (Simple logic to choose Regressor or Classifier)
# Assuming if target is numeric and many unique values -> Regression
# Else -> Classification
# This is a heuristic for code generation
if df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 10:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    metric_print = "print(f'MSE: {mean_squared_error(y_test, y_pred)}')"
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    metric_print = "print(f'Accuracy: {accuracy_score(y_test, y_pred)}')"

# Pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
eval(metric_print)
"""
    return code
