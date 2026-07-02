"""
RAG knowledge base: builds a small corpus of role/skill "advice" documents
(why a skill matters, how to build it, common role expectations) and indexes
them with FAISS for retrieval-grounded feedback generation.

This knowledge base is intentionally structured/curated (not scraped) so
retrieval results are trustworthy and citeable -- this is what makes the
downstream feedback "explainable": every piece of advice traces back to a
specific retrieved knowledge chunk.
"""
import json
import numpy as np
import faiss
from dataclasses import dataclass
from src.embeddings import EmbeddingBackend


# Short, factual "why this matters / how to build it" notes per skill.
# Not exhaustive -- covers the most common skills across the skill bank.
SKILL_NOTES = {
    "python": "Python is a widely used general-purpose language valued for readability and its large ecosystem (data, web, automation). Build proficiency via small scripting projects, then a framework like Flask/Django.",
    "sql": "SQL is the standard language for querying and manipulating relational databases. Practice writing joins, aggregations, and window functions on sample datasets (e.g. Kaggle datasets, LeetCode SQL problems).",
    "java": "Java is a mainstay of enterprise backend systems, prized for its portability and mature tooling. Strengthen it by building a small REST service with Spring Boot.",
    "javascript": "JavaScript is essential for web interactivity and increasingly for backend (Node.js). Practice via building small interactive front-end components.",
    "react": "React is the dominant library for building component-based web UIs. Build a small multi-page app with state management to demonstrate proficiency.",
    "aws": "AWS is the most widely adopted cloud platform. Hands-on experience with EC2, S3, and IAM (via the free tier) demonstrates practical cloud fluency.",
    "docker": "Docker containerizes applications for consistent deployment. Practice by containerizing a personal project end-to-end (Dockerfile + docker-compose).",
    "kubernetes": "Kubernetes orchestrates containerized workloads at scale. A local cluster (minikube/kind) is a good way to practice deployments, services, and scaling.",
    "machine learning": "Machine learning skills are demonstrated through applied projects: data cleaning, model selection, evaluation, and clear communication of results -- not just theory.",
    "data science": "Data science roles value the full pipeline: exploratory analysis, feature engineering, modeling, and communicating insights to non-technical stakeholders.",
    "agile": "Agile/Scrum experience signals comfort working in iterative, cross-functional teams. Mentioning specific ceremonies (standups, retros, sprint planning) strengthens this.",
    "git": "Version control (Git) is a baseline expectation for almost all technical roles. A public GitHub history with meaningful commits is strong evidence.",
    "spring": "Spring (and Spring Boot) is the standard framework for enterprise Java applications, handling dependency injection, web layers, and data access.",
    "django": "Django is a batteries-included Python web framework, popular for rapid, secure web application development.",
    "tensorflow": "TensorFlow is a major deep learning framework used for building and deploying neural networks in production.",
    "pytorch": "PyTorch is the leading deep learning framework in research and increasingly in production, valued for its flexibility.",
    "tableau": "Tableau is a leading data visualization tool for building interactive dashboards for business stakeholders.",
    "excel": "Advanced Excel (pivot tables, formulas, basic modeling) remains a baseline analytical tool expected in many data/business roles.",
    "linux": "Linux systems administration knowledge (shell, permissions, processes) underlies most server-side and DevOps work.",
    "terraform": "Terraform enables infrastructure-as-code, letting teams version and automate cloud infrastructure provisioning.",
    "selenium": "Selenium is a standard tool for browser-based test automation, key for QA/Testing roles.",
    "figma": "Figma is the industry-standard collaborative UI design tool for UI/UX roles.",
    "network security": "Network security expertise covers protecting infrastructure via firewalls, VPNs, and monitoring -- often validated through certifications (Security+, CCNA Security).",
    "etl": "ETL (Extract-Transform-Load) skills involve building reliable data pipelines that move and clean data between systems.",
    "stored procedures": "Stored procedures are precompiled SQL routines used for performance and encapsulating business logic inside the database layer.",
}

ROLE_EXPECTATION_TEMPLATE = (
    "For a {role} role, employers typically look for a combination of the core "
    "required skills, demonstrated through real projects or work history, plus "
    "evidence of applying them in a team/production context (not just familiarity)."
)


@dataclass
class KBDoc:
    doc_id: str
    text: str
    metadata: dict


def build_knowledge_base(skill_bank: dict) -> list[KBDoc]:
    docs = []
    # one doc per (role, note) pair we have factual info for
    for role, skills in skill_bank.items():
        docs.append(KBDoc(
            doc_id=f"role::{role}",
            text=ROLE_EXPECTATION_TEMPLATE.format(role=role) + f" Core skills: {', '.join(skills)}.",
            metadata={"type": "role_overview", "role": role},
        ))
        for skill in skills:
            if skill in SKILL_NOTES:
                docs.append(KBDoc(
                    doc_id=f"skill::{role}::{skill}",
                    text=f"[{role}] {skill}: {SKILL_NOTES[skill]}",
                    metadata={"type": "skill_note", "role": role, "skill": skill},
                ))
    return docs


class RAGIndex:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend
        self.docs: list[KBDoc] = []
        self.index = None

    def build(self, docs: list[KBDoc]):
        self.docs = docs
        texts = [d.text for d in docs]
        embeddings = self.backend.encode(texts).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine since embeddings are normalized
        self.index.add(embeddings)
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[KBDoc, float]]:
        q_emb = self.backend.encode([query]).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.docs[idx], float(score)))
        return results


if __name__ == "__main__":
    from src.data_prep import load_skill_bank
    bank = load_skill_bank()
    docs = build_knowledge_base(bank)
    print(f"Built {len(docs)} knowledge base documents.")

    backend = EmbeddingBackend()
    backend.fit([d.text for d in docs])
    rag = RAGIndex(backend).build(docs)

    results = rag.retrieve("missing python and machine learning skills for data science role", top_k=3)
    for doc, score in results:
        print(f"[{score:.3f}] {doc.doc_id}: {doc.text[:100]}")
