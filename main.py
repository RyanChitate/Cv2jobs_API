from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model_loader import load_model
from utils import extract_text_from_pdf, preprocess_text
from job_matcher import match_jobs
import pandas as pd

app = FastAPI(title="CV2Impact API", description="AI-Powered CV Matching", version="1.0")
model = load_model()

@app.post("/match")
async def match_cv(cv: UploadFile = File(...), jobs: UploadFile = File(...)):
    try:
        # Extract and clean CV
        cv_text = extract_text_from_pdf(cv.file)
        cv_cleaned = preprocess_text(cv_text)

        # Read job descriptions
        job_df = pd.read_csv(jobs.file)
        job_descriptions = job_df["description"].dropna().tolist()

        # Run similarity matching
        ranked_jobs = match_jobs(cv_cleaned, job_descriptions, model)

        return JSONResponse(content={"matches": ranked_jobs})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
