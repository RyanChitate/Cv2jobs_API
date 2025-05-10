# 🌍 CV2Impact API – AI-Powered Resume Matching System

**CV2Impact API** is a backend service designed to analyze resumes (CVs) and intelligently match them to job descriptions using advanced Natural Language Processing (NLP). It is part of the `CareerCompass` suite developed by Team Delta0, focused on empowering African youth through ethical AI and career-driven technology.

This FastAPI-based system transforms resume data into actionable insights, offering:
- Job fit ranking (based on semantic similarity)
- Upskilling recommendations (based on skill gaps)
- CSV-ready and JSON-friendly outputs
- Internal CLI or future API-driven integration with private dashboards

The system is optimized for internal use within private teams or policy-driven institutions aiming to tackle youth unemployment and digital exclusion in the 4IR era.

## 🔧 Key Features

- 🧠 Resume understanding using `SentenceTransformer` models
- 📊 Job ranking with normalized fit scores (0–100%)
- 📈 JSON and CSV outputs for reports or dashboards
- 🔁 Extendable design for educational, NGO, and government pilot programs

## 🚀 Tech Stack

- FastAPI (backend)
- PyPDF2 (CV parsing)
- Pandas (data handling)
- SentenceTransformers (NLP embeddings)
- Scikit-learn (cosine similarity)

## 🔐 Use Case

This project is intended for:
- Educational institutions running career support programs
- Government or NGO partners working on employment initiatives
- AI-for-Good teams building localized solutions for skills development

---

