from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes import dataset, eda, preprocessing, pca, regression, classification, unsupervised, ensemble
from utils.helpers import clean_old_sessions

app = FastAPI(
    title="IPL Machine Learning Exploratory Data Analysis (EDA) REST API",
    description="A modular REST API backend providing IPL dataset cleaning, analysis, transformations, and PCA reduction.",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(dataset.router)
app.include_router(eda.router)
app.include_router(preprocessing.router)
app.include_router(pca.router)
app.include_router(regression.router)
app.include_router(classification.router)
app.include_router(unsupervised.router)
app.include_router(ensemble.router)

@app.on_event("startup")
async def startup_event():
    """Startup task to initialize clean temporary folders."""
    print("Initializing server temp directory...")
    clean_old_sessions()
    print("Server ready.")

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "IPL ML LAB EDA API Service",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
