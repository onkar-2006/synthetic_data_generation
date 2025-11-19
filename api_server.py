import pandas as pd
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
# IMPORTANT: Import CORSMiddleware to fix the cross-origin fetch errors
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

# --- Core Modules from your project ---
# NOTE: These imports assume you have 'core/graph_state.py' and 'main_graph.py'
from core.graph_state import GenerationState
from main_graph import build_generation_graph

# --- FastAPI Setup ---
app = FastAPI(title="Synthetic Data Generation API", version="1.0.0")
app_graph = build_generation_graph()

# --- CORS Configuration ---
# Fixes the 'Access-Control-Allow-Origin' CORS policy error when testing with a local HTML file.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- State Management ---
LATEST_STATUS = {"status": "Awaiting Run", "quality_score": None, "error_message": None, "project_id": "P_001"}
SYNTHETIC_DATA = pd.DataFrame()
UPLOAD_DIR = "uploads"


# Define the models for endpoint responses
class PipelineStatus(BaseModel):
    status: str
    quality_score: float | None
    error_message: str | None
    synthetic_row_count: int


class SyntheticDataResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]


class FileUploadResponse(BaseModel):
    file_path: str
    message: str


# --- Utility Function to Load Data ---
def load_latest_synthetic_data(project_id: str):
    """Attempts to load the last generated CSV data into the global DataFrame based on project_id."""
    global SYNTHETIC_DATA
    global LATEST_STATUS

    file_path = f"synthetic_data_{project_id}.csv"

    if not os.path.exists(file_path):
        SYNTHETIC_DATA = pd.DataFrame()
        return

    try:
        SYNTHETIC_DATA = pd.read_csv(file_path)
        LATEST_STATUS['project_id'] = project_id
    except Exception as e:
        SYNTHETIC_DATA = pd.DataFrame()


# --- Endpoint 0: Upload File ---
@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file_handler(file: UploadFile = File(...)):
    """Accepts a file upload and saves it to the local filesystem."""

    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file on server: {e}")
    finally:
        await file.close()

    return FileUploadResponse(file_path=file_path, message=f"File '{file.filename}' uploaded successfully.")


# --- Endpoint 1: Run the Pipeline ---
@app.post("/run-pipeline", response_model=PipelineStatus)
async def run_data_pipeline(
        input_file_path: str,
        project_id: str = "P_001"
):
    """
    Triggers the LangGraph pipeline to generate synthetic data.
    """
    global LATEST_STATUS
    global SYNTHETIC_DATA

    if not os.path.exists(input_file_path):
        raise HTTPException(
            status_code=400,
            detail=f"Input file not found at path: {input_file_path}. Please check the path."
        )

    initial_state = GenerationState(
        project_id=project_id,
        input_file_path=input_file_path,
        original_data=pd.DataFrame(),
        inferred_schema={},
        user_constraints=[],
        synthetic_data=None,
        quality_report={},
        status='Initialized',
        log_messages=[],
        error_message=None
    )

    try:
        final_state = app_graph.invoke(initial_state)

        LATEST_STATUS['status'] = final_state['status']
        LATEST_STATUS['error_message'] = final_state.get('error_message')
        LATEST_STATUS['quality_score'] = final_state.get('quality_report', {}).get('Overall Score')
        LATEST_STATUS['project_id'] = project_id

        if LATEST_STATUS['status'] == 'Quality Approved':
            load_latest_synthetic_data(project_id)
        else:
            SYNTHETIC_DATA = pd.DataFrame()

    except Exception as e:
        error_msg = f"CRITICAL GRAPH EXECUTION ERROR: {e}"
        LATEST_STATUS.update({
            "status": "Error",
            "error_message": error_msg,
            "quality_score": None
        })
        raise HTTPException(status_code=500, detail=error_msg)

    return PipelineStatus(
        status=LATEST_STATUS['status'],
        quality_score=LATEST_STATUS['quality_score'],
        error_message=LATEST_STATUS['error_message'],
        synthetic_row_count=SYNTHETIC_DATA.shape[0]
    )


# --- Endpoint 2: Get Current Status ---
@app.get("/status", response_model=PipelineStatus)
def get_status(project_id: str = "P_001"):
    """Returns the status and quality score of the last pipeline run for the given project_id."""
    if project_id != LATEST_STATUS['project_id']:
        return PipelineStatus(
            status="Awaiting Run",
            quality_score=None,
            error_message=None,
            synthetic_row_count=0
        )

    return PipelineStatus(
        status=LATEST_STATUS['status'],
        quality_score=LATEST_STATUS['quality_score'],
        error_message=LATEST_STATUS['error_message'],
        synthetic_row_count=SYNTHETIC_DATA.shape[0]
    )


# --- Endpoint 3: Retrieve Synthetic Data ---
@app.get("/data", response_model=SyntheticDataResponse)
def get_synthetic_data(project_id: str = "P_001"):
    """Returns the last generated synthetic data in a JSON structure for the given project_id."""

    if project_id != LATEST_STATUS['project_id'] or SYNTHETIC_DATA.empty:
        load_latest_synthetic_data(project_id)

    if SYNTHETIC_DATA.empty:
        raise HTTPException(status_code=404,
                            detail=f"No synthetic data available for project {project_id}. Run the pipeline first.")

    data_list = SYNTHETIC_DATA.to_dict('records')

    return SyntheticDataResponse(
        columns=SYNTHETIC_DATA.columns.tolist(),
        data=data_list
    )


# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    """Loads existing synthetic data (default project) and ensures upload directory exists when the server starts."""
    load_latest_synthetic_data(LATEST_STATUS['project_id'])
    os.makedirs(UPLOAD_DIR, exist_ok=True)

