from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import shutil
import os
import uuid
from api.services import eda, modeling

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_FILE_PATH = {}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() != "csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_location = f"{UPLOAD_DIR}/{file_id}.csv"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    CURRENT_FILE_PATH["latest"] = file_location
    
    return {"message": "File uploaded successfully", "file_id": file_id, "filename": file.filename}

@router.post("/analyze")
async def analyze_data():
    file_path = CURRENT_FILE_PATH.get("latest")
    if not file_path:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    result = eda.analyze_dataframe(file_path)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/plot")
async def plot_data():
    file_path = CURRENT_FILE_PATH.get("latest")
    if not file_path:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    plots = eda.generate_plots(file_path)
    return plots

@router.post("/modeling")
async def modeling_code(target: str = Form(None)):
    file_path = CURRENT_FILE_PATH.get("latest")
    if not file_path:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    code = modeling.generate_model_code(file_path, target)
    return {"code": code}
