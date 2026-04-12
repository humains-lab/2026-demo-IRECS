import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys

# Force-prioritize the local 'code' directory
CODE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code'))
if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

# Verify locations
try:
    import g3p
    import main
    import job_manager
    print(f"DEBUG: g3p from {g3p.__file__}")
    print(f"DEBUG: main from {main.__file__}")
    # print(f"DEBUG: job_manager from {job_manager.__file__}")
except Exception as e:
    print(f"DEBUG: Import error during verification: {e}")

from job_manager import manager

app = FastAPI(title="IRECS API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets'))
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../uploads'))

os.makedirs(UPLOAD_DIR, exist_ok=True)

class ExperimentConfig(BaseModel):
    datasetFilePath: str
    grammarFilePath: Optional[str] = "assets/ml.xml"
    nFolds: Optional[int] = 2
    maxGenerations: Optional[int] = 10
    populationSize: Optional[int] = 30
    crossProb: Optional[float] = 0.9
    mutationProb: Optional[float] = 0.1
    fitnessThreshold: Optional[float] = 0.4
    replacementStrategy: Optional[str] = "NEWPOPULATION"
    classificationStrategy: Optional[str] = "CBA"
    positiveWeight: Optional[float] = 1.5
    seed: Optional[int] = 1
    vocabStrategy: Optional[str] = "ALL"
    extraTerms: Optional[str] = ""

@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    # List from datasets/
    if os.path.exists(DATASETS_DIR):
        for f in os.listdir(DATASETS_DIR):
            if f.endswith(".csv"):
                datasets.append({"name": f, "path": os.path.join(DATASETS_DIR, f), "source": "system"})
    
    # List from uploads/
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(".csv"):
                datasets.append({"name": f, "path": os.path.join(UPLOAD_DIR, f), "source": "upload"})
                
    return datasets

import io
import pandas as pd
import logging

def enrich_with_scopus(df: pd.DataFrame, api_key: Optional[str] = None):
    """
    Enriches the dataframe with Scopus metadata if columns are missing.
    Uses the logic from datasetCrawler.py
    """
    enrich_cols = ["nCites", "authorCount", "citedby_count", "aggregationType"]
    # Check for presence of at least the main metrics (independent of case)
    df_cols_lower = [c.lower() for c in df.columns]
    missing_enrich = [col for col in enrich_cols if col.lower() not in df_cols_lower]
    
    # Also check if they exist but are mostly empty
    for col in ["nCites", "authorCount", "aggregationType"]:
        col_name = next((c for c in df.columns if c.lower() == col.lower()), None)
        if col_name and df[col_name].isnull().all():
            if col not in missing_enrich:
                missing_enrich.append(col)

    if not missing_enrich:
        return df
        
    try:
        from pybliometrics.scopus import ScopusSearch
        import os
        
        # Inject the key into environment immediately
        if api_key:
            os.environ['PYBLIOMETRICS_API_KEY'] = api_key
        
        # Check if we have an API key (either in env or if config exists)
        has_local_config = os.path.exists(os.path.expanduser("~/.scopus/config.ini"))
        has_env_key = os.environ.get('PYBLIOMETRICS_API_KEY')
        
        if not has_local_config and not has_env_key:
             return df # No key, no enrichment, but no crash
            
        logging.info(f"Validating Scopus API Key...")
        
        # TEST THE KEY with a dummy search before proceeding
        try:
            # We use a very specific query that should return 0 or 1 result but verify auth
            ScopusSearch('DOI(10.1016/j.jss.2016.02.044)')
        except Exception as auth_err:
            if "401" in str(auth_err) or "Unauthorized" in str(auth_err):
                raise HTTPException(
                    status_code=401, 
                    detail=f"Scopus Authentication Failed: The API Key provided is invalid (Unauthorized). "
                           f"Please check your key at https://dev.elsevier.com/"
                )
            logging.warning(f"Scopus validation gave an unexpected error: {auth_err}")
            return df

        logging.info(f"Scopus key validated. Enriching dataset with missing columns: {missing_enrich}")
        
        # Add missing columns
        for col in missing_enrich:
            df[col] = "N/A"
            
        for index, row in df.iterrows():
            doi = str(row.get('doi', ''))
            if doi and doi not in ['[]', '-1', 'N/A', 'undetermined', 'nan']:
                try:
                    s = ScopusSearch(doi)
                    if s.results and len(s.results) > 0:
                        res = s.results[0]
                        if "nCites" in missing_enrich: df.at[index, 'nCites'] = getattr(res, 'citedby_count', 0)
                        if "authorCount" in missing_enrich: df.at[index, 'authorCount'] = getattr(res, 'author_count', 0)
                        if "citedby_count" in missing_enrich: df.at[index, 'citedby_count'] = getattr(res, 'citedby_count', 0)
                        if "aggregationType" in missing_enrich: df.at[index, 'aggregationType'] = getattr(res, 'aggregationType', 'unknown')
                except Exception as e:
                    logging.warning(f"Failed to query Scopus for DOI {doi}: {e}")
                    
        return df
    except ImportError:
        logging.error("pybliometrics not installed. Skipping enrichment.")
        return df
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Scopus enrichment failed: {e}")
        return df

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...), 
    scopus_api_key: Optional[str] = None
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        content = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Detect if enrichment is needed
        df = pd.read_csv(file_path)
        has_doi = any(c.lower() == "doi" for c in df.columns)
        has_title = any(c.lower() in ["document title", "title"] for c in df.columns)
        
        if has_doi and not has_title:
            # Start background enrichment
            task_id = manager.create_enrichment_task(file.filename)
            manager.start_enrichment(task_id, file_path, api_key=scopus_api_key)
            return {"task_id": task_id, "status": "enriching"}
        
        return {"name": file.filename, "path": file_path, "status": "completed"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.get("/api/enrichment/status/{task_id}")
async def get_enrichment_status(task_id: str):
    if task_id not in manager.enrichments:
        raise HTTPException(status_code=404, detail="Task not found")
    return manager.enrichments[task_id]

@app.post("/api/experiment/run")
async def run_experiment(config: ExperimentConfig):
    job_id = manager.create_job(config.model_dump())
    manager.start_job(job_id)
    return {"job_id": job_id}

@app.get("/api/experiment/status/{job_id}")
async def get_status(job_id: str):
    status = manager.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.post("/api/experiment/pause/{job_id}")
async def pause_experiment(job_id: str):
    manager.toggle_pause(job_id)
    return {"status": "success"}

@app.post("/api/experiment/update/{job_id}")
async def update_experiment_config(job_id: str, config: ExperimentConfig):
    manager.update_job_config(job_id, config.model_dump())
    return {"status": "success"}

@app.get("/api/experiment/results/{job_id}")
async def get_results(job_id: str):
    status = manager.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return {
        "results": status["results"],
        "best_rules": status["best_rules"],
        "logs": status["logs"],
        "fitness_history": status["fitness_history"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
