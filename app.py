import gradio as gr, PyPDF2, os, re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

SKILL_BANK = ["Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning",
              "LLM", "Llama-3", "LoRA", "PEFT", "Transformers", "RAG", "Qdrant",
              "FastAPI", "Gradio", "AI Agents", "Colab", "SQL", "Docker", "Kubernetes"]

def extract_text(path):
    try:
        text = " ".join([(p.extract_text() or "") for p in PyPDF2.PdfReader(path).pages])
        return text
    except:
        return ""

def analyze_resume(jd, files):
    if not jd or not files:
        return [], "Please paste JD and upload resumes"

    jd_emb = model.encode([jd])
    results = []

    for f in files:
        resume_text = extract_text(f)
        if not resume_text:
            continue

        # Match Score
        res_emb = model.encode([resume_text])
        score = float(cosine_similarity(jd_emb, res_emb)[0][0]) * 100

        # Strengths & Missing
        found_skills = [s for s in SKILL_BANK if s.lower() in resume_text.lower()]
        jd_skills = [s for s in SKILL_BANK if s.lower() in jd.lower()]
        missing = [s for s in jd_skills if s not in found_skills]

        file_name = os.path.basename(f)

        results.append([
            file_name,
            f"{score:.1f}%",
            ", ".join(found_skills[:6]) if found_skills else "General",
            ", ".join(missing[:5]) if missing else "None - Strong Match",
            "Hire" if score > 75 else "Review" if score > 55 else "Reject"
        ])

    # Sort by score
    results = sorted(results, key=lambda x: float(x[1].replace('%','')), reverse=True)

    return results, f"✅ Screened {len(results)} resumes successfully!"

with gr.Blocks(title="AI Resume Screener - GenAI Phi3 RAG") as demo:
    gr.Markdown("# 🚀 AI Resume Screener - Bulk Screening with Explainable AI")
    gr.Markdown("Upload 10+ resumes, get Match Score, Strengths, Missing Skills automatically")

    with gr.Row():
        jd_input = gr.Textbox(label="Paste Job Description", lines=8, placeholder="Paste JD here...")
        file_input = gr.File(label="Upload Resumes (PDF - Multiple)", file_count="multiple", file_types=[".pdf"])

    btn = gr.Button("🔍 Screen Resumes", variant="primary")

    with gr.Row():
        output_table = gr.Dataframe(
            headers=["Resume", "Match Score", "Strengths", "Missing Skills", "Decision"],
            label="Results - Sorted by Best Match"
        )

    status = gr.Textbox(label="Status")

    btn.click(fn=analyze_resume, inputs=[jd_input, file_input], outputs=[output_table, status])

demo.launch()
