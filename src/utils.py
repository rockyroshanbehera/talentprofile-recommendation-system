"""
utils.py - Helper Functions and Formatting Utilities
TalentProfile AI - Universal Talent Recommendation System
"""

import os
import pandas as pd
from typing import List, Tuple

def get_filter_options(talent_profiles: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract sorted, unique filter options from the real dataset without inventing categories.
    """
    # Top industries (filter out blanks)
    ind_counts = talent_profiles["industry_name"].value_counts()
    top_industries = [ind for ind, count in ind_counts.items() if str(ind).strip() and count >= 50]
    industries = ["All Industries"] + sorted(top_industries)

    # Experience levels
    exp_counts = talent_profiles["formatted_experience_level"].value_counts()
    experiences = ["All Experience Levels"] + [exp for exp, count in exp_counts.items() if str(exp).strip()]

    # Work types
    wt_counts = talent_profiles["formatted_work_type"].value_counts()
    work_types = ["All Work Types"] + [wt for wt, count in wt_counts.items() if str(wt).strip()]

    return industries, experiences, work_types

def get_dataset_summary(talent_profiles: pd.DataFrame) -> dict:
    return {
        "total_profiles": len(talent_profiles),
        "unique_industries": int(talent_profiles["industry_name"].nunique()),
        "unique_skills": int(talent_profiles["skill_name"].nunique()),
        "experience_levels": int(talent_profiles["formatted_experience_level"].nunique()),
    }
