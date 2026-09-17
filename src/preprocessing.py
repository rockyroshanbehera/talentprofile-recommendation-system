"""
preprocessing.py - Text Cleaning and Feature Preprocessing Module
TalentProfile AI - Universal Talent Recommendation System
"""

import re
from typing import List, Set
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS: Set[str] = set(ENGLISH_STOP_WORDS)
EXTRA_STOPWORDS = {"need", "looking", "want", "hire", "seeking", "find", "search", "please"}
EXTENDED_STOP_WORDS = STOP_WORDS.union(EXTRA_STOPWORDS)

def clean_text(text: str, remove_recruiter_stopwords: bool = False) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    active_stopwords = EXTENDED_STOP_WORDS if remove_recruiter_stopwords else STOP_WORDS
    words = [
        word for word in text.split()
        if word not in active_stopwords and len(word) > 1
    ]
    return " ".join(words)

def extract_query_keywords(query: str) -> List[str]:
    cleaned = clean_text(query, remove_recruiter_stopwords=True)
    return cleaned.split()
