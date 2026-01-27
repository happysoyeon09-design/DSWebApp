# Data Analysis Web App

This is a web application for automated Exploratory Data Analysis (EDA) and Machine Learning Code Generation.

## Structure
- **Backend**: FastAPI (Port 8000) - Handles data processing, plotting, and code generation.
- **Frontend**: Flask (Port 5000) - Serves the web interface.

## Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run

1. **Start the Backend (FastAPI)**
   Open a terminal and run:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

2. **Start the Frontend (Flask)**
   Open a NEW terminal and run:
   ```bash
   cd web
   python app.py
   ```

3. **Access the App**
   Open your browser and go to: `http://127.0.0.1:5000`

## Features
1. **Upload CSV**: Upload a dataset for analysis.
2. **Analysis**: View basic statistics (Head, Describe, Missing Values).
3. **Visualization**: See correlation heatmaps and histograms.
4. **Modeling**: Select a target variable and generate Python code for training a Random Forest model.
