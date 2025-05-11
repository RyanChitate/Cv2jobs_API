from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import pandas as pd
import re

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ===== Init App =====
app = FastAPI(
    title="CV2Impact API",
    description="AI-powered CV-to-Job matching engine for internal use.",
    version="1.0"
)

# ===== Enable CORS for Swagger UI testing =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load SentenceTransformer Model =====
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ===== Utils =====
def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(cv_text, job_text):
    cv_vec = model.encode([cv_text])
    job_vec = model.encode([job_text])
    return cosine_similarity(cv_vec, job_vec)[0][0]

# ===== Main API Route =====
@app.post("/match")
async def match_cv(cv: UploadFile = File(...), jobs: UploadFile = File(...)):
    try:
        # 1. Extract CV text
        cv_raw = extract_text_from_pdf(cv.file)
        cv_cleaned = preprocess_text(cv_raw)

        # 2. Load job descriptions
        job_df = pd.read_csv(jobs.file)
        if "description" not in job_df.columns:
            return JSONResponse(status_code=400, content={"error": "CSV must have a 'description' column."})

        job_descriptions = job_df["description"].dropna().tolist()

        # 3. Calculate similarity for each job
        results = []
        max_score = 0
        scores = []

        for job in job_descriptions:
            job_cleaned = preprocess_text(job)
            score = calculate_similarity(cv_cleaned, job_cleaned)
            scores.append(score)
            if score > max_score:
                max_score = score

        # 4. Normalize and format results
        for i, score in enumerate(scores):
            relative = (score / max_score) * 100 if max_score > 0 else 0
            results.append({
                "job_snippet": job_descriptions[i].strip().split('\n')[0][:80],
                "raw_score": round(score, 4),
                "relative_score_percent": round(relative, 2)
            })

        return JSONResponse(content={"matches": results})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
