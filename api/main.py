from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import processing

app = FastAPI(title="Data Analysis API", description="API for Auto-EDA and Modeling", version="1.0")

# CORS Configuration
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "*"  # For development simplicity
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(processing.router, prefix="/api", tags=["processing"])

@app.get("/")
def read_root():
    return {"message": "Data Analysis API is running"}
