# Resume Evaluation AI

An end-to-end system that evaluates candidate resumes against job
requirements using ML/embedding-based matching, generates personalized
explainable feedback through a retrieval-grounded (RAG) generation
pipeline, and offers a conversational interface for candidates.

## Structure

```
resume_evaluation_ai/
├── resumes_dataset.jsonl          # source dataset (3,500 resumes, 36 categories)
├── skill_bank.json                # curated required-skills-per-role bank
├── src/
│   ├── data_prep.py                # loading, cleaning, skill extraction
│   ├── embeddings.py               # embedding backend (sentence-transformers + offline TF-IDF/SVD fallback)
│   ├── matcher.py                  # resume<->role match scoring + skill-gap analysis
│   ├── baseline_classifier.py      # supervised TF-IDF+LogReg baseline (comparison only)
│   ├── rag_knowledge.py            # curated knowledge base + FAISS retrieval index
│   └── feedback_generator.py       # Ollama client + RAG-grounded feedback generation
├── 01_data_prep_and_matching.ipynb # Part 1: EDA, skill bank, matching engine, evaluation
├── 02_rag_feedback_and_chat.ipynb  # Part 2: RAG knowledge base, feedback generation, chat overview
├── app.py                          # Gradio conversational app
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Both notebooks were executed and verified to run cleanly end-to-end in a
sandboxed environment **without** access to `huggingface.co` (for
sentence-transformer model downloads) or `localhost:11434` (Ollama). In
that environment, the pipeline automatically falls back to:
- **TF-IDF + Truncated SVD** embeddings instead of `sentence-transformers`
- **Template-based feedback generation** instead of Ollama LLM calls

**On your own machine**, if you have internet access and/or Ollama
running, both integrations activate automatically — no code changes
needed. To use Ollama:

```bash
ollama serve
ollama pull gemma4:e4b   # or your preferred model — update OLLAMA_MODEL in src/feedback_generator.py
```

## Running the notebooks

Open `01_data_prep_and_matching.ipynb` first (data prep, skill bank,
matching engine, evaluation), then `02_rag_feedback_and_chat.ipynb`
(RAG knowledge base, explainable feedback, chat overview). Both notebooks
expect `resumes_dataset.jsonl`.
## Running the conversational app

```bash
python app.py
```

Opens a local Gradio UI where you can **upload a PDF resume directly**
(auto-extracted via `pypdf`, with a `pdfplumber` fallback for trickier
layouts) or paste resume text manually, pick a target role, get a match
score + explainable feedback, and ask follow-up questions in a chat
interface. If a PDF is a scanned image with no text layer, the app will
warn you and suggest pasting the text instead (OCR is not included, since
resumes are essentially always text-based PDFs).

## Key design decisions

- **Skill bank instead of a JD dataset**: the source data has no job
  description field, so required skills per role are either mined from
  real resume text (29/36 categories) or manually curated where the
  mined data was unreliable (7/36 categories — see notebook 1, Section 2
  for details on which categories and why).
- **Offline-safe by default**
- **Explainability**