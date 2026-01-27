from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for simplicity in this demo (in prod, use DB or session)
data_store = {"df": None}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    data_store["df"] = df.fillna(0) # Simple handling for demo
    return {
        "message": "File uploaded successfully", 
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "head": df.head().to_dict(orient="records")
    }

@app.get("/analyze")
def analyze():
    df = data_store["df"]
    if df is None:
        return {"error": "No data uploaded"}
    
    desc = df.describe().to_dict()
    missing = df.isnull().sum().to_dict()
    return {"description": desc, "missing": missing}

@app.get("/visualize")
def visualize():
    df = data_store["df"]
    if df is None:
        return {"error": "No data uploaded"}
    
    # Generate a correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    images = []
    
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        images.append({"title": "Correlation Matrix", "data": img_str})

        # Generate histograms for first 3 numeric columns
        for col in numeric_df.columns[:3]:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            images.append({"title": f"Distribution of {col}", "data": img_str})

    return {"images": images}

@app.post("/model_code")
def generate_model_code(target: str = Form(...)):
    df = data_store["df"]
    if df is None:
        return {"error": "No data uploaded"}
    
    # Determine task type based on target column
    if target not in df.columns:
        return {"error": f"Target column '{target}' not found"}
    
    is_numeric = pd.api.types.is_numeric_dtype(df[target])
    task_type = "Regression" if is_numeric else "Classification"
    
    full_code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForest{'Regressor' if task_type == 'Regression' else 'Classifier'}
from sklearn.metrics import {'mean_squared_error, r2_score' if task_type == 'Regression' else 'accuracy_score, classification_report'}

# 1. Load Data
# Assuming 'data.csv' is your file
df = pd.read_csv('data.csv')

# 2. Preprocessing
# Handle missing values
df = df.fillna(0)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 3. Features and Target
target = '{target}'
X = df.drop(columns=[target])
y = df[target]

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeling ({task_type})
model = RandomForest{'Regressor' if task_type == 'Regression' else 'Classifier'}(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
"""
    if task_type == 'Regression':
        full_code += """
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
"""
    else:
        full_code += """
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
"""

    return {"code": full_code, "task_type": task_type}
