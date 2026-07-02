"""
Ollama client wrapper for grounded feedback generation.

This is designed to be run on machine where Ollama is actually
reachable (http://localhost:11434) -- the sandboxed environment used to
build this project cannot reach your local network, so this client can't
be used end-to-end here.

If Ollama is unreachable (e.g. not running), `generate_feedback()`
transparently falls back to a template-based generator - so the pipeline always produces a usable result.
"""
import json
import os
import requests
from src.matcher import MatchResult
from src.rag_knowledge import RAGIndex

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"  # change to whatever you have pulled, e.g. "llama3", "mistral"
# Default generation timeout. The FIRST call to a model is often slow (Ollama
# has to load the full model into memory/VRAM before generating anything --
# this alone can take 30-90+ seconds on CPU-only setups). Subsequent calls
# are much faster since the model stays loaded. Override with the
# OLLAMA_TIMEOUT env var if you still see read timeouts.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 120.0))


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL, timeout: float = OLLAMA_TIMEOUT):
        self.host = host
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def warm_up(self) -> bool:
        """Sends a trivial generate call to force Ollama to load the model into
        memory now, rather than on the user's first real request (which is
        where the read-timeout usually happens). Uses a long timeout since
        cold-start loading can be slow; safe to call once at app startup."""
        try:
            self.generate("hi", system="Reply with one word.")
            return True
        except Exception as e:
            print(f"[ollama] warm-up call failed or timed out ({e}); "
                  f"the first real request may be slow while the model loads.")
            return False

    def generate(self, prompt: str, system: str = None) -> str:
        """Calls Ollama's /api/generate endpoint (non-streaming)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    def chat(self, messages: list[dict]) -> str:
        """Calls Ollama's /api/chat endpoint. messages: [{"role": "user"/"assistant"/"system", "content": "..."}]"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


FEEDBACK_SYSTEM_PROMPT = (
    "You are a helpful, honest career coach giving a candidate feedback on how well "
    "their resume matches a target job role. You are given: (1) a quantitative match "
    "analysis, and (2) grounded knowledge-base notes about the missing skills. "
    "Write concise, encouraging, specific feedback (150-250 words). Reference the "
    "actual matched/missing skills. Do not invent skills or facts not given to you."
)


def build_feedback_prompt(match: MatchResult, retrieved_notes: list) -> str:
    notes_text = "\n".join(f"- {doc.text}" for doc, score in retrieved_notes) if retrieved_notes else "(no additional notes retrieved)"
    prompt = f"""Candidate match analysis for the role of {match.role}:
- Overall match score: {match.overall_score}/100
- Semantic fit: {match.semantic_score}/100
- Skill coverage: {match.skill_overlap_score}/100
- Matched skills: {', '.join(match.matched_skills) or 'none'}
- Missing skills: {', '.join(match.missing_skills) or 'none'}
- Additional skills candidate has beyond this role's core list: {', '.join(match.extra_skills[:8]) or 'none'}

Relevant knowledge base notes on the missing skills:
{notes_text}

Write personalized feedback for this candidate."""
    return prompt


def template_feedback(match: MatchResult, retrieved_notes: list) -> str:
    """Deterministic, fully offline fallback -- grounded in the same data an LLM would see."""
    lines = []
    lines.append(f"**Match summary for {match.role}:** {match.overall_score}/100 overall "
                  f"(semantic fit {match.semantic_score}/100, skill coverage {match.skill_overlap_score}/100).")

    if match.matched_skills:
        lines.append(f"\n**Strengths:** Your resume shows {len(match.matched_skills)} of the core skills "
                      f"expected for this role: {', '.join(match.matched_skills)}. This is a solid foundation.")

    if match.missing_skills:
        lines.append(f"\n**Gaps to close:** {len(match.missing_skills)} core skills weren't detected in your "
                      f"resume: {', '.join(match.missing_skills)}.")
        if retrieved_notes:
            lines.append("Here's why they matter and how to start building them:")
            for doc, score in retrieved_notes[:4]:
                if doc.metadata.get("type") == "skill_note":
                    lines.append(f"  - {doc.text.split(': ', 1)[-1]}")

    if match.extra_skills:
        lines.append(f"\n**Bonus:** You also show skills outside this role's core list "
                      f"({', '.join(match.extra_skills[:5])}), which could be worth highlighting if relevant.")

    lines.append(f"\n**Bottom line:** {'Strong match -- ' if match.overall_score >= 60 else 'Room to grow -- '}"
                  f"focus on closing the highest-impact skill gaps above, and make sure your resume text "
                  f"explicitly names skills you already have, since automated screens look for exact matches.")

    return "\n".join(lines)


def generate_feedback(match: MatchResult, rag_index: RAGIndex, ollama_client: OllamaClient = None,
                       top_k_notes: int = 5) -> dict:
    """Main entry point: retrieves grounding notes, then generates feedback via
    Ollama if available, else falls back to the template generator.
    Returns dict with 'feedback', 'source' ('ollama' | 'template'), and 'retrieved_notes'."""
    query = f"missing skills {', '.join(match.missing_skills)} for {match.role} role"
    retrieved = rag_index.retrieve(query, top_k=top_k_notes)

    client = ollama_client or OllamaClient()
    if client.is_available():
        try:
            prompt = build_feedback_prompt(match, retrieved)
            text = client.generate(prompt, system=FEEDBACK_SYSTEM_PROMPT)
            if text:
                return {"feedback": text, "source": "ollama", "retrieved_notes": retrieved}
        except Exception as e:
            print(f"[feedback] Ollama call failed ({e}); falling back to template.")

    return {"feedback": template_feedback(match, retrieved), "source": "template", "retrieved_notes": retrieved}


if __name__ == "__main__":
    from src.data_prep import load_skill_bank, load_and_clean
    from src.matcher import ResumeMatcher
    from src.rag_knowledge import build_knowledge_base
    from src.embeddings import EmbeddingBackend

    bank = load_skill_bank()
    df = load_and_clean("/mnt/user-data/uploads/resumes_dataset.jsonl")

    matcher = ResumeMatcher(bank)
    matcher.fit(df["clean_text"].tolist())

    kb_docs = build_knowledge_base(bank)
    rag_backend = EmbeddingBackend()
    rag_backend.fit([d.text for d in kb_docs])
    rag = RAGIndex(rag_backend).build(kb_docs)

    sample = df[df["Category"] == "Data Science"].iloc[5]
    match = matcher.match(sample["Text"], "Data Science")
    result = generate_feedback(match, rag)
    print(f"[source: {result['source']}]\n")
    print(result["feedback"])
