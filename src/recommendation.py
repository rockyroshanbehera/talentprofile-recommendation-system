"""
recommendation.py - Talent Profile Recommendation and Ranking Engine
TalentProfile AI - Universal Talent Recommendation System
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.preprocessing import clean_text, extract_query_keywords
from src.model_downloader import ensure_model_files_exist

class TalentRecommender:
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        self.models_dir = models_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.talent_profiles = None
        self.is_loaded = False

    def load_models(self) -> None:
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(self.models_dir, "tfidf_matrix.pkl")
        prof_path = os.path.join(self.models_dir, "talent_profiles.pkl")

        # 1. Verify lightweight vectorizer in repository
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Missing vectorizer model file at: {vec_path}")

        # 2. Automatically check and fetch large artifacts from GitHub Releases if missing
        ensure_model_files_exist(self.models_dir)

        # 3. Load pre-trained vectorizer and matrix representations
        self.vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = joblib.load(mat_path)
        self.talent_profiles = joblib.load(prof_path)
        self.is_loaded = True

    def explain_match(self, query_tokens: List[str], profile_row: pd.Series) -> Dict[str, Any]:
        """
        Generate transparent skill overlap and matched keywords breakdown.
        Distinguishes overall similarity score from exact keyword/skill overlap.
        """
        if not query_tokens:
            return {"matched_skills": [], "matched_keywords": [], "overlap_count": 0}

        profile_skills_str = str(profile_row.get("skill_name", "")).lower()
        profile_title_str = str(profile_row.get("title", "")).lower()
        profile_desc_str = str(profile_row.get("description", ""))[:1000].lower()

        matched_skills = []
        matched_keywords = []

        # Check skills column
        known_skills = [s.strip() for s in profile_skills_str.split(",") if s.strip()]
        for skill in known_skills:
            for token in query_tokens:
                if token in skill:
                    matched_skills.append(skill.title())
                    break

        # Check title & description for tokens
        for token in query_tokens:
            pattern = rf"\b{re.escape(token)}\b"
            if re.search(pattern, profile_title_str) or re.search(pattern, profile_desc_str):
                matched_keywords.append(token.title())

        # Deduplicate
        matched_skills = sorted(list(set(matched_skills)))
        matched_keywords = sorted(list(set(matched_keywords)))

        return {
            "matched_skills": matched_skills,
            "matched_keywords": matched_keywords,
            "overlap_count": len(matched_skills) + len(matched_keywords)
        }

    def recommend(
        self,
        query: str,
        top_n: int = 10,
        industry_filter: Optional[str] = None,
        experience_filter: Optional[str] = None,
        work_type_filter: Optional[str] = None,
        location_keyword: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            self.load_models()

        if not query or not query.strip():
            return []

        cleaned_q = clean_text(query, remove_recruiter_stopwords=False)
        if not cleaned_q:
            return []

        query_vec = self.vectorizer.transform([cleaned_q])
        
        # Sparse dot product for cosine similarity (matrix rows & query vector are L2 normalized)
        sim_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()

        # Apply filtering if requested
        candidate_indices = np.arange(len(self.talent_profiles))

        if industry_filter and industry_filter != "All Industries":
            ind_mask = (self.talent_profiles["industry_name"].astype(str) == industry_filter).values
            candidate_indices = candidate_indices[ind_mask[candidate_indices]]

        if experience_filter and experience_filter != "All Experience Levels":
            exp_mask = (self.talent_profiles["formatted_experience_level"].astype(str) == experience_filter).values
            candidate_indices = candidate_indices[exp_mask[candidate_indices]]

        if work_type_filter and work_type_filter != "All Work Types":
            wt_mask = (self.talent_profiles["formatted_work_type"].astype(str) == work_type_filter).values
            candidate_indices = candidate_indices[wt_mask[candidate_indices]]

        if location_keyword and location_keyword.strip():
            loc_lower = location_keyword.strip().lower()
            loc_mask = self.talent_profiles["location"].str.lower().str.contains(loc_lower, na=False, regex=False).values
            candidate_indices = candidate_indices[loc_mask[candidate_indices]]

        if len(candidate_indices) == 0:
            return []

        # Filter candidate scores
        candidate_sims = sim_scores[candidate_indices]
        
        # Apply min similarity threshold if given
        if min_similarity > 0:
            valid_mask = candidate_sims >= min_similarity
            candidate_indices = candidate_indices[valid_mask]
            candidate_sims = candidate_sims[valid_mask]

        if len(candidate_indices) == 0:
            return []

        # Sort top N
        sorted_rel_order = candidate_sims.argsort()[::-1][:top_n]
        top_actual_indices = candidate_indices[sorted_rel_order]
        top_scores = candidate_sims[sorted_rel_order]

        query_tokens = extract_query_keywords(query)

        recommendations = []
        for rank, (idx, score) in enumerate(zip(top_actual_indices, top_scores), start=1):
            row = self.talent_profiles.iloc[idx]
            explanation = self.explain_match(query_tokens, row)
            
            raw_desc = str(row.get("description", ""))
            short_desc = raw_desc[:320] + "..." if len(raw_desc) > 320 else raw_desc
            
            rec = {
                "rank": rank,
                "job_id": row.get("job_id"),
                "title": row.get("title", "Unknown Title"),
                "company_name": row.get("company_name", "N/A"),
                "industry_name": row.get("industry_name", "General"),
                "skill_name": row.get("skill_name", "General / Not Specified"),
                "formatted_experience_level": row.get("formatted_experience_level", "Not Specified"),
                "formatted_work_type": row.get("formatted_work_type", "Not Specified"),
                "location": row.get("location", "Remote / Unspecified"),
                "short_description": short_desc,
                "full_description": raw_desc,
                "similarity_score": float(score),
                "similarity_percentage": round(float(score) * 100, 1),
                "matched_skills": explanation["matched_skills"],
                "matched_keywords": explanation["matched_keywords"],
                "overlap_count": explanation["overlap_count"]
            }
            recommendations.append(rec)

        return recommendations
