import os
import sys
try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop
    sys.modules['audioop'] = audioop

import gradio as gr
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_pdf_text(f):
    try:
        reader = PyPDF2.PdfReader(f)
        return "".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def analyze_resume(jd, files):
    if not jd or not files:
        return [], "Upload JD and resumes"
    results = []
    for fo in files:
        txt = extract_pdf_text(fo.name)
        if not txt.strip(): continue
        vec = TfidfVectorizer(stop_words='english').fit_transform([jd, txt])
        score = __import__('sklearn.metrics.pairwise', fromlist=['cosine_similarity']).cosine_similarity(vec[0:1], vec[1:2])[0][0]*100
        fn = os.path.basename(fo.name)
        results.append([fn, f"{score:.1f}%", "Matched keywords", "Check JD", "Shortlist" if score>60 else "Review"])
    results.sort(key=lambda x: float(x[1].replace('%','')), reverse=True)
    return results, f"Done {len(results)} resumes"

with gr.Blocks(title="AI Resume Screener") as demo:
    gr.Markdown("# AI Resume Screener")
    jd_input = gr.Textbox(label="Job Description", lines=8)
    file_input = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
    btn = gr.Button("Screen Resumes", variant="primary")
    out = gr.Dataframe(headers=["Resume","Score","Strengths","Missing","Decision"])
    status = gr.Textbox(label="Status")
    btn.click(fn=analyze_resume, inputs=[jd_input, file_input], outputs=[out, status])

# THIS IS THE ONLY LAUNCH LINE - MUST BE LAST
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
