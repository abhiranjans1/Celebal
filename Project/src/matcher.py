"""
Resume <-> Job Role matching engine.

Combines:
  1. Semantic similarity (embedding cosine similarity between resume text
     and a synthesized "role profile" text built from the curated skill bank)
  2. Explicit skill overlap (extracted_skills vs required skills for the role)

into a single explainable match score + a breakdown of matched/missing skills.
"""
from dataclasses import dataclass, field
from src.embeddings import EmbeddingBackend, similarity
from src.data_prep import extract_skills, normalize_text


@dataclass
class MatchResult:
    role: str
    overall_score: float          # 0-100
    semantic_score: float         # 0-100, embedding-based
    skill_overlap_score: float    # 0-100, explicit skill coverage
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    extra_skills: list = field(default_factory=list)  # candidate has, role doesn't require

    def summary(self) -> str:
        lines = [
            f"Role: {self.role}",
            f"Overall match score: {self.overall_score:.1f}/100",
            f"  - Semantic similarity: {self.semantic_score:.1f}/100",
            f"  - Skill coverage:      {self.skill_overlap_score:.1f}/100",
            f"Matched skills ({len(self.matched_skills)}): {', '.join(self.matched_skills) or 'none'}",
            f"Missing skills ({len(self.missing_skills)}): {', '.join(self.missing_skills) or 'none'}",
        ]
        return "\n".join(lines)


def build_role_profile_text(role: str, skill_bank: dict) -> str:
    """Synthesize a lightweight 'job description' from curated required skills."""
    skills = skill_bank.get(role, [])
    return f"{role} requiring skills in " + ", ".join(skills)


@dataclass
class CandidateRankResult:
    """One ranked candidate for a free-text job description search."""
    row_index: int
    overall_score: float
    semantic_score: float
    skill_overlap_score: float
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # e.g. Category, Source, Name-safe fields


class ResumeMatcher:
    def __init__(self, skill_bank: dict, embedding_backend: EmbeddingBackend = None):
        self.skill_bank = skill_bank
        self.backend = embedding_backend or EmbeddingBackend()
        self._role_profiles = {r: build_role_profile_text(r, skill_bank) for r in skill_bank}
        self._role_embeddings = None
        self._roles_order = list(self._role_profiles.keys())
        self._full_vocab = sorted({s for skills in skill_bank.values() for s in skills})

        # candidate pool cache, populated by fit_candidate_pool()
        self._candidate_df = None
        self._candidate_embeddings = None

    def fit(self, resume_corpus: list[str]):
        """Fit the embedding backend (needed for tfidf-svd fallback) on a corpus
        that should include both resume texts AND role profile texts, so the
        vocabulary covers both sides."""
        full_corpus = list(resume_corpus) + list(self._role_profiles.values())
        self.backend.fit(full_corpus)
        self._role_embeddings = self.backend.encode(list(self._role_profiles.values()))
        return self

    def match(self, resume_text: str, target_role: str) -> MatchResult:
        if target_role not in self.skill_bank:
            raise ValueError(f"Unknown role '{target_role}'. Known roles: {list(self.skill_bank)}")

        required_skills = self.skill_bank[target_role]

        # 1. explicit skill overlap
        candidate_skills = extract_skills(resume_text, required_skills)
        # also compute the candidate's full skill set against the whole vocab for "extra skills"
        candidate_all_skills = extract_skills(resume_text, self._full_vocab)

        matched = [s for s in required_skills if s in candidate_skills]
        missing = [s for s in required_skills if s not in candidate_skills]
        extra = [s for s in candidate_all_skills if s not in required_skills]

        skill_overlap_score = 100.0 * len(matched) / max(1, len(required_skills))

        # 2. semantic similarity
        role_idx = self._roles_order.index(target_role)
        resume_emb = self.backend.encode([resume_text])
        role_emb = self._role_embeddings[role_idx:role_idx + 1]
        sem = float(similarity(resume_emb, role_emb)[0][0])
        # cosine sim in [-1,1] (or [0,1] for tfidf/normalized) -> rescale to 0-100
        semantic_score = max(0.0, min(1.0, (sem + 1) / 2 if sem < 0 else sem)) * 100.0

        overall = 0.5 * semantic_score + 0.5 * skill_overlap_score

        return MatchResult(
            role=target_role,
            overall_score=round(overall, 1),
            semantic_score=round(semantic_score, 1),
            skill_overlap_score=round(skill_overlap_score, 1),
            matched_skills=matched,
            missing_skills=missing,
            extra_skills=extra[:15],
        )

    def best_matching_roles(self, resume_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Rank all roles by overall match score for a given resume (career-fit discovery)."""
        results = []
        for role in self._roles_order:
            m = self.match(resume_text, role)
            results.append((role, m.overall_score))
        return sorted(results, key=lambda x: -x[1])[:top_k]

    # ------------------------------------------------------------------
    # Free-text job description support: score one resume against an
    # arbitrary pasted JD (not limited to the curated skill-bank roles),
    # and rank a whole candidate pool against a JD.
    # ------------------------------------------------------------------

    def extract_jd_requirements(self, jd_text: str) -> list:
        """Skills detected in a free-text job description, against the full
        known skill vocabulary (union across all curated roles)."""
        return extract_skills(jd_text, self._full_vocab)

    def match_against_jd(self, resume_text: str, jd_text: str, jd_required_skills: list = None) -> MatchResult:
        """Score a single resume against an arbitrary free-text job description
        (rather than a curated role). Required skills are auto-extracted from
        the JD text unless explicitly provided."""
        jd_required = jd_required_skills if jd_required_skills is not None else self.extract_jd_requirements(jd_text)

        candidate_skills = extract_skills(resume_text, jd_required) if jd_required else []
        candidate_all_skills = extract_skills(resume_text, self._full_vocab)

        matched = [s for s in jd_required if s in candidate_skills]
        missing = [s for s in jd_required if s not in candidate_skills]
        extra = [s for s in candidate_all_skills if s not in jd_required]

        skill_overlap_score = 100.0 * len(matched) / max(1, len(jd_required)) if jd_required else 0.0

        resume_emb = self.backend.encode([resume_text])
        jd_emb = self.backend.encode([jd_text])
        sem = float(similarity(resume_emb, jd_emb)[0][0])
        semantic_score = max(0.0, min(1.0, (sem + 1) / 2 if sem < 0 else sem)) * 100.0

        overall = 0.5 * semantic_score + 0.5 * skill_overlap_score

        return MatchResult(
            role="Custom Job Description",
            overall_score=round(overall, 1),
            semantic_score=round(semantic_score, 1),
            skill_overlap_score=round(skill_overlap_score, 1),
            matched_skills=matched,
            missing_skills=missing,
            extra_skills=extra[:15],
        )

    def fit_candidate_pool(self, candidates_df, text_col: str = "Text", clean_text_col: str = "clean_text",
                            metadata_cols: list = None):
        """Precompute and cache embeddings + extracted skills for a pool of
        candidate resumes (e.g. the whole dataset), so repeated JD searches
        against the same pool don't re-embed thousands of resumes each time.
        Call once at startup; `rank_candidates_for_jd()` reuses the cache."""
        metadata_cols = metadata_cols or []
        self._candidate_df = candidates_df.reset_index(drop=True).copy()
        self._candidate_embeddings = self.backend.encode(self._candidate_df[clean_text_col].tolist())

        if "extracted_skills" not in self._candidate_df.columns:
            self._candidate_df["extracted_skills"] = self._candidate_df[clean_text_col].apply(
                lambda t: extract_skills(t, self._full_vocab)
            )
        self._candidate_text_col = text_col
        self._candidate_metadata_cols = metadata_cols
        return self

    def rank_candidates_for_jd(self, jd_text: str, top_k: int = 10) -> tuple:
        """Given a free-text job description, rank the cached candidate pool
        (see fit_candidate_pool) by match score. Returns
        (jd_required_skills, list[CandidateRankResult])."""
        if self._candidate_embeddings is None:
            raise RuntimeError("Call fit_candidate_pool(df) before rank_candidates_for_jd().")

        jd_required = self.extract_jd_requirements(jd_text)
        jd_emb = self.backend.encode([jd_text])
        sims = similarity(self._candidate_embeddings, jd_emb).flatten()

        results = []
        for i, row in self._candidate_df.iterrows():
            cand_skills = row["extracted_skills"]
            matched = [s for s in jd_required if s in cand_skills]
            missing = [s for s in jd_required if s not in cand_skills]
            skill_score = 100.0 * len(matched) / max(1, len(jd_required)) if jd_required else 0.0

            sim = float(sims[i])
            sem_score = max(0.0, min(1.0, (sim + 1) / 2 if sim < 0 else sim)) * 100.0
            overall = 0.5 * sem_score + 0.5 * skill_score

            metadata = {col: row[col] for col in self._candidate_metadata_cols if col in row}
            results.append(CandidateRankResult(
                row_index=int(i),
                overall_score=round(overall, 1),
                semantic_score=round(sem_score, 1),
                skill_overlap_score=round(skill_score, 1),
                matched_skills=matched,
                missing_skills=missing,
                metadata=metadata,
            ))

        results.sort(key=lambda r: -r.overall_score)
        return jd_required, results[:top_k]
