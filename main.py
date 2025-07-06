from fastapi import FastAPI, Depends, HTTPException,UploadFile,File,BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Tuple, Callable, TypeVar
import os 
import logging

# Custom Modules
from src.config import APP_NAME, VERSION, API_SECRET_KEY, DOWNLOADED_IMAGES_FOLDER
from src.schemas import PredictionsResponses
from src.inference import ImageClassifier

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Initialize classifier
classifier=ImageClassifier()

# Initialize an app
app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.info(f"App started: {APP_NAME} v{VERSION}")

# API key verification
api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        logging.warning("Unauthorized API access attempt")
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key

@app.get('/', tags=['check'])
async def home(api_key: str=Depends(verify_api_key)):
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "status": "up & running"
    }

@app.post("/classify-batch-memory", tags=['CNN'], response_model=PredictionsResponses)
async def classify_batch_memory(files: List[UploadFile] = File(...), api_key: str = Depends(verify_api_key)):
    """Classify multiple images in a batch and images are stored in memory (for small sizes batches)"""
    try:
        logging.info(f"In-memory classification: {len(files)} files")

        if not files:
            raise HTTPException(400, "No files provided")
        
        # Validate all files are images
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(400, f"File {file.filename} is not an image")
            
        # Read all files
        contents = []
        for file in files:
            contents.append(await file.read())

        # Call the function 
        predictions = classifier.predict_batch(images_data=contents)
        return predictions
    
    except Exception as e:
        logging.error(f"Error in memory classification: {str(e)}")
        raise HTTPException(500, f"Error making predictions: {str(e)}")
    

def delete_files(file_paths: List[str]):
    """Delete the specified files"""
    logging.info(f"Cleaning up {len(file_paths)} files")
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")


@app.post("/classify-batch-paths", tags=['CNN'], response_model=PredictionsResponses)
async def classify_batch_paths(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), api_key: str = Depends(verify_api_key)):
    """Classify multiple images in a batch and images are saved on disk (for big sizes batches)"""
    try:
        logging.info(f"Classification: {len(files)} files")

        if not files:
            raise HTTPException(400, "No files provided")
        
        
        saved_paths = []
        # Save files to disk
        
        for file in files:
            # Check if it's an image
            if not file.content_type.startswith("image/"):
                raise HTTPException(400, f"File {file.filename} is not an image")
            # Save with original basename
            file_path=os.path.join(DOWNLOADED_IMAGES_FOLDER,os.path.basename(file.filename))
                        # Save file to disk
            with open(file_path, "wb") as f:
                f.write(await file.read())

            saved_paths.append(file_path)

        # Classify the saved images
        predictions = classifier.predict_batch(saved_paths)

        # Add background task to clean up files
        background_tasks.add_task(delete_files, saved_paths)
        return predictions
        
    except Exception as e:
        logging.error(f"Error in file-based classification: {str(e)}")
        raise HTTPException(500, f"Error processing images: {str(e)}")
        






