import pandas as pd
import io
import json
import seaborn as sns
import matplotlib.pyplot as plt
import base64

def analyze_dataframe(file_path: str):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": str(e)}

    # Basic Info
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient="records"),
        "describe": df.describe().to_dict()
    }
    return summary

def generate_plots(file_path: str):
    df = pd.read_csv(file_path)
    plots = {}
    
    # Select numeric columns for correlation heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Histograms for Numeric Columns (limit to first 5 to avoid overload)
        for col in numeric_df.columns[:5]:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots[f'hist_{col}'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
    return plots
