from sklearn.metrics.pairwise import cosine_similarity

def match_jobs(cv_text, job_descriptions, model):
    scores = []
    cv_vec = model.encode([cv_text])

    for job in job_descriptions:
        job_vec = model.encode([job])
        score = cosine_similarity(cv_vec, job_vec)[0][0]
        scores.append(score)

    normalized = [(s / max(scores)) * 100 if max(scores) > 0 else 0 for s in scores]
    
    return [
        {"job_description": job.strip().split('\n')[0][:80], "score": round(s, 2)}
        for job, s in zip(job_descriptions, normalized)
    ]
