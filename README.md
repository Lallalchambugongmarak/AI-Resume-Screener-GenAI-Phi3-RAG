# 🤖 AI Resume Screener - GenAI | Phi-3 + RAG + Qdrant

> Bulk ATS that ranks Top 10 candidates with Match % + Strengths + Missing Skills. Explainable AI for HR.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/YOUR_USERNAME_HERE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![GenAI](https://img.shields.io/badge/GenAI-Phi3--mini--4k-green)]()

## 🚀 Problem Solved
HR gets 500+ resumes per JD. Manual screening = 20+ hours. This tool does it in 2 minutes with AI reasoning.

## 🧠 Tech Stack (Recruiter Keywords)
- **LLM:** Phi-3-mini-4k (4-bit quantized), Llama-3
- **RAG:** Sentence-Transformers + Qdrant Vector DB
- **Framework:** FastAPI + Gradio
- **Parsing:** PyPDF2
- **Deployment:** Hugging Face Spaces + Docker

## ✨ Features
- Upload 50+ PDFs at once
- Paste JD + Get Ranked Top 10
- Each candidate: Match Score %, Key Strengths, Skill Gaps
- Explainable AI - Why selected?

## 📊 How it Works
1. Parse resumes -> Chunk -> Embed (all-MiniLM-L6-v2)
2. Store in Qdrant
3. JD embedding -> Similarity search
4. Phi-3 reasons: Strengths + Missing Skills
5. Gradio UI shows leaderboard

## 🏃 Run Locally
pip install -r requirements.txt
python app.py

## 👨‍💻 Author
Lallal Chambugong Marak - GenAI Engineer | Open to Fresher AI Roles | Bangalore
