"""
Resume Evaluation AI -- Conversational Interface (Gradio)

Run locally with:
    python app.py

Requires Ollama running locally (http://localhost:11434) with the model
set in src/feedback_generator.py (default: gemma4:e4b) for LLM-generated
chat responses. Falls back to template-based responses if Ollama isn't
reachable, so the app still works out of the box.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.data_prep import load_skill_bank, load_and_clean
from src.embeddings import EmbeddingBackend
from src.matcher import ResumeMatcher
from src.rag_knowledge import build_knowledge_base, RAGIndex
from src.feedback_generator import OllamaClient, generate_feedback, FEEDBACK_SYSTEM_PROMPT
from src.pdf_reader import extract_resume_text

DATA_PATH = os.environ.get("RESUME_DATA_PATH", "resumes_dataset.jsonl")

print("Loading pipeline (this fits the model once at startup)...")
skill_bank = load_skill_bank()
df = load_and_clean(DATA_PATH)

matcher = ResumeMatcher(skill_bank)
matcher.fit(df["clean_text"].tolist())

print("Indexing candidate pool for JD-based search (one-time embedding pass)...")
matcher.fit_candidate_pool(df, metadata_cols=["Category", "Source"])
print(f"Candidate pool indexed: {len(df)} resumes.")

kb_docs = build_knowledge_base(skill_bank)
rag_backend = EmbeddingBackend()
rag_backend.fit([d.text for d in kb_docs])
rag = RAGIndex(rag_backend).build(kb_docs)

ollama = OllamaClient()
ROLES = sorted(skill_bank.keys())

ollama_available = ollama.is_available()
if ollama_available:
    print(f"Ollama reachable. Warming up model '{ollama.model}' (loads into memory now, "
          f"so your first real question isn't slow)...")
    ollama.warm_up()
    print("Model warm-up done.")
else:
    print("Ollama not reachable at startup -- app will use offline template feedback. "
          "Start `ollama serve` and restart this app to enable LLM-generated responses.")

print(f"Pipeline ready. Ollama reachable: {ollama_available}")

# Module-level state: this app is designed for local single-user use, so a
# plain dict avoids the gr.State/ChatInterface closure-sync pitfalls.
_session_state = {"resume_text": "", "last_match": None}
_jd_session_state = {"jd_text": "", "jd_required": [], "ranked": []}


def on_pdf_upload(file_obj):
    """Extracts text from an uploaded PDF and fills the resume textbox."""
    if file_obj is None:
        return gr.update(), "No file uploaded."

    path = file_obj if isinstance(file_obj, str) else file_obj.name
    result = extract_resume_text(path)

    if result["warning"]:
        status = f"{result['warning']} (extracted {len(result['text'])} chars via {result['method']})"
    else:
        status = f"Extracted {len(result['text'])} characters from PDF (method: {result['method']})"

    return gr.update(value=result["text"]), status


def search_candidates_for_jd(jd_text: str, top_k: int):
    if not jd_text or not jd_text.strip():
        return "Please paste a job description first.", ""

    jd_required, ranked = matcher.rank_candidates_for_jd(jd_text, top_k=int(top_k))

    if not jd_required:
        req_note = "No recognized skills were detected in this JD text (matching is running on semantic similarity only, which is less precise). Try including more specific skill/tool names."
    else:
        req_note = f"**Detected required skills ({len(jd_required)}):** {', '.join(jd_required)}"

    rows = ["| Rank | Score | Category | Source | Matched Skills | Missing Skills |",
            "|---|---|---|---|---|---|"]
    for rank, r in enumerate(ranked, start=1):
        cat = r.metadata.get("Category", "?")
        src = r.metadata.get("Source", "?")
        matched = ", ".join(r.matched_skills) or "—"
        missing = ", ".join(r.missing_skills) or "—"
        rows.append(f"| {rank} | {r.overall_score} | {cat} | {src} | {matched} | {missing} |")

    table_md = "\n".join(rows)

    _jd_session_state["jd_text"] = jd_text
    _jd_session_state["jd_required"] = jd_required
    _jd_session_state["ranked"] = ranked

    return req_note, table_md


def run_evaluation(resume_text: str, target_role: str):
    if not resume_text or not resume_text.strip():
        return "Please paste a resume first.", None
    if not target_role:
        return "Please select a target role.", None

    match = matcher.match(resume_text, target_role)
    result = generate_feedback(match, rag, ollama_client=ollama)

    score_md = (
        f"### Match score: {match.overall_score}/100\n"
        f"- Semantic fit: {match.semantic_score}/100\n"
        f"- Skill coverage: {match.skill_overlap_score}/100\n\n"
        f"**Matched:** {', '.join(match.matched_skills) or 'none'}\n\n"
        f"**Missing:** {', '.join(match.missing_skills) or 'none'}\n\n"
        f"*(feedback generated via: {result['source']})*"
    )

    _session_state["resume_text"] = resume_text
    _session_state["last_match"] = match

    return score_md, result["feedback"]


# Conversational layer
def chat_respond(message, history):
    match = _session_state.get("last_match")

    if match is None:
        return "Please run an evaluation first (paste your resume + pick a role above), then ask me anything about your results."

    # retrieve fresh context relevant to the user's specific question
    retrieved = rag.retrieve(message + " " + match.role, top_k=4)
    context = "\n".join(f"- {doc.text}" for doc, score in retrieved)

    system = FEEDBACK_SYSTEM_PROMPT + (
        " You are now in a follow-up conversation with the candidate about their "
        "existing match results below. Answer their specific question."
    )
    prompt = f"""Candidate's match analysis for {match.role}:
- Overall score: {match.overall_score}/100
- Matched skills: {', '.join(match.matched_skills) or 'none'}
- Missing skills: {', '.join(match.missing_skills) or 'none'}

Relevant knowledge notes:
{context}

Candidate's question: {message}"""

    if ollama.is_available():
        try:
            reply = ollama.generate(prompt, system=system)
            if reply:
                return reply
        except Exception as e:
            print(f"[chat] Ollama failed: {e}")

    # offline fallback: simple grounded templated reply
    fallback = (
        f"(Ollama not reachable, using offline mode)\n\n"
        f"Based on your {match.role} match ({match.overall_score}/100): "
        f"your missing skills are {', '.join(match.missing_skills) or 'none'}. "
        f"Relevant notes:\n{context or 'No additional notes found.'}"
    )
    return fallback


# ---- JD search follow-up chat ----
def jd_chat_respond(message, history):
    ranked = _jd_session_state.get("ranked")
    jd_required = _jd_session_state.get("jd_required") or []

    if not ranked:
        return "Please search for candidates first (paste a JD + click Search above), then ask me anything about the results."

    # summarize top candidates for grounding (cap to keep prompt manageable)
    top_n = ranked[:10]
    candidates_summary = "\n".join(
        f"- Rank {i+1}: {r.metadata.get('Category','?')} candidate (row {r.row_index}), "
        f"score {r.overall_score}/100, matched: {', '.join(r.matched_skills) or 'none'}, "
        f"missing: {', '.join(r.missing_skills) or 'none'}"
        for i, r in enumerate(top_n)
    )

    retrieved = rag.retrieve(message + (" " + " ".join(jd_required) if jd_required else ""), top_k=4)
    context = "\n".join(f"- {doc.text}" for doc, score in retrieved)

    system = (
        "You are a helpful, honest recruiting assistant. You are given a ranked list of "
        "candidates for a job description, plus grounded knowledge-base notes. Answer the "
        "recruiter's specific question about these candidates concisely and factually. "
        "Do not invent candidates, scores, or skills not given to you."
    )
    prompt = f"""Job description requirements detected: {', '.join(jd_required) if jd_required else 'none detected'}

Ranked candidates:
{candidates_summary}

Relevant knowledge notes:
{context}

Recruiter's question: {message}"""

    if ollama.is_available():
        try:
            reply = ollama.generate(prompt, system=system)
            if reply:
                return reply
        except Exception as e:
            print(f"[jd_chat] Ollama failed: {e}")

    # offline fallback: grounded templated reply
    fallback = (
        f"(Ollama not reachable, using offline mode)\n\n"
        f"Here are the top candidates for this search:\n{candidates_summary}\n\n"
        f"Relevant notes:\n{context or 'No additional notes found.'}"
    )
    return fallback


with gr.Blocks(title="Resume Evaluation AI") as demo:
    gr.Markdown("# Resume Evaluation AI")

    with gr.Tabs():
        with gr.Tab("Evaluate a resume against a role"):
            gr.Markdown("Upload a PDF resume (or paste text), pick a target role, and get an explainable match score, skill-gap feedback, and a chat interface to ask follow-up questions.")

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.File(label="Upload resume PDF (optional)", file_types=[".pdf"], type="filepath")
                    pdf_status = gr.Markdown("")
                    resume_input = gr.Textbox(label="Resume text", lines=15, placeholder="Paste resume text here, or upload a PDF above to auto-fill...")
                    role_dropdown = gr.Dropdown(choices=ROLES, label="Target role", value=ROLES[0] if ROLES else None)
                    evaluate_btn = gr.Button("Evaluate", variant="primary")
                with gr.Column(scale=2):
                    score_output = gr.Markdown(label="Score")
                    feedback_output = gr.Markdown(label="Feedback")

            pdf_input.change(
                on_pdf_upload,
                inputs=[pdf_input],
                outputs=[resume_input, pdf_status],
            )

            evaluate_btn.click(
                run_evaluation,
                inputs=[resume_input, role_dropdown],
                outputs=[score_output, feedback_output],
            )

            gr.Markdown("---\n## Ask follow-up questions")
            chatbot = gr.ChatInterface(fn=chat_respond)

        with gr.Tab("Find candidates for a job description"):
            gr.Markdown(
                f"Paste a custom job description and search the existing candidate pool "
                f"({len(df)} resumes) for the best matches. Required skills are auto-detected "
                f"from your JD text; each candidate is scored on the same semantic fit + skill "
                f"coverage basis as the single-resume evaluation above."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    jd_input = gr.Textbox(
                        label="Job description",
                        lines=12,
                        placeholder="Paste a full job description here (responsibilities, required skills, etc.)...",
                    )
                    top_k_slider = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of candidates to return")
                    search_btn = gr.Button("Search candidates", variant="primary")
                with gr.Column(scale=3):
                    jd_requirements_output = gr.Markdown(label="Detected requirements")
                    candidates_output = gr.Markdown(label="Ranked candidates")

            search_btn.click(
                search_candidates_for_jd,
                inputs=[jd_input, top_k_slider],
                outputs=[jd_requirements_output, candidates_output],
            )

            gr.Markdown("---\n## Ask follow-up questions")
            jd_chatbot = gr.ChatInterface(fn=jd_chat_respond)

if __name__ == "__main__":
    demo.launch()
