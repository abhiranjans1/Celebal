"""
Data preparation and skill-extraction utilities for the Resume Evaluation AI system.

Handles:
  - Loading & cleaning the raw resumes_dataset.jsonl
  - Regex-based skill extraction from free text (robust to the fact that the
    'Skills' column is a templated placeholder for a large chunk of records)
"""
import json
import re
import pandas as pd
from pathlib import Path

SKILL_BANK_PATH = Path(__file__).parent.parent / "skill_bank.json"


def load_skill_bank() -> dict:
    with open(SKILL_BANK_PATH) as f:
        return json.load(f)


def build_full_skill_vocab(skill_bank: dict) -> list:
    """Union of all skills across all categories, deduped, for global extraction."""
    vocab = set()
    for skills in skill_bank.values():
        vocab.update(skills)
    return sorted(vocab)


def normalize_text(text: str) -> str:
    """Lowercase, strip decorative/unicode symbols (emoji headers seen in synthetic
    resumes), collapse whitespace, but keep characters relevant to tech skills
    (+, #, ., /) so tokens like 'c++', 'c#', 'node.js', 'ci/cd' survive."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # drop emoji / symbol glyphs and non-ascii decorative characters
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    # keep alnum + a few skill-relevant punctuation marks
    text = re.sub(r"[^a-z0-9\+\#\./\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_SKILL_PATTERN_CACHE = {}


def _pattern_for(skill: str) -> re.Pattern:
    if skill not in _SKILL_PATTERN_CACHE:
        escaped = re.escape(skill)
        _SKILL_PATTERN_CACHE[skill] = re.compile(r"\b" + escaped + r"\b")
    return _SKILL_PATTERN_CACHE[skill]


def extract_skills(text: str, vocab: list) -> list:
    """Extract which skills (from vocab) appear in the given text via
    word-boundary regex matching. Returns skills in the order they appear
    in `vocab` (stable ordering)."""
    norm = normalize_text(text)
    found = []
    for skill in vocab:
        if _pattern_for(skill).search(norm):
            found.append(skill)
    return found


def load_and_clean(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["clean_text"] = df["Text"].apply(normalize_text)
    df["text_len"] = df["clean_text"].str.len()
    return df


def enrich_with_extracted_skills(df: pd.DataFrame, skill_bank: dict) -> pd.DataFrame:
    """Adds an 'extracted_skills' column: skills found in the resume text,
    restricted to the full known vocabulary (so we don't pick up noise)."""
    vocab = build_full_skill_vocab(skill_bank)
    df = df.copy()
    df["extracted_skills"] = df["clean_text"].apply(lambda t: extract_skills(t, vocab))
    df["n_extracted_skills"] = df["extracted_skills"].apply(len)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/resumes_dataset.jsonl"
    bank = load_skill_bank()
    df = load_and_clean(path)
    df = enrich_with_extracted_skills(df, bank)
    print(df[["Category", "Source", "n_extracted_skills"]].groupby("Category").mean(numeric_only=True))
