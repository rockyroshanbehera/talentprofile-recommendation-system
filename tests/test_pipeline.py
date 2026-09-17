"""
test_pipeline.py - Automated validation suite for TalentProfile AI using standard unittest
"""

import unittest
import os
from src.preprocessing import clean_text, extract_query_keywords
from src.recommendation import TalentRecommender
from src.utils import get_filter_options, get_dataset_summary

class TestTalentProfileAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = TalentRecommender()
        cls.recommender.load_models()

    def test_01_preprocessing(self):
        text = "<p>Looking for a <b>Python</b> Data Analyst with SQL & AWS! http://example.com</p>"
        cleaned = clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertIn("python", cleaned)
        self.assertIn("data", cleaned)
        self.assertIn("analyst", cleaned)
        self.assertIn("sql", cleaned)
        self.assertIn("aws", cleaned)

    def test_02_extract_query_keywords(self):
        tokens = extract_query_keywords("Need a machine learning engineer with Python")
        self.assertIn("machine", tokens)
        self.assertIn("learning", tokens)
        self.assertIn("engineer", tokens)
        self.assertIn("python", tokens)
        self.assertNotIn("need", tokens)

    def test_03_model_loading(self):
        self.assertTrue(self.recommender.is_loaded)
        self.assertIsNotNone(self.recommender.vectorizer)
        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertIsNotNone(self.recommender.talent_profiles)
        self.assertGreater(len(self.recommender.talent_profiles), 100000)

    def test_04_empty_query(self):
        results = self.recommender.recommend("")
        self.assertEqual(results, [])
        results2 = self.recommender.recommend("   ")
        self.assertEqual(results2, [])

    def test_05_top_k_variations(self):
        q = "Python data analyst with SQL and Power BI experience"
        for k in [5, 10, 20]:
            results = self.recommender.recommend(q, top_n=k)
            self.assertLessEqual(len(results), k)
            self.assertGreater(len(results), 0)
            scores = [r["similarity_score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_06_explainable_matching(self):
        q = "Need a Python data analyst with SQL and Power BI experience"
        results = self.recommender.recommend(q, top_n=5)
        for r in results:
            self.assertIn("rank", r)
            self.assertIn("similarity_percentage", r)
            self.assertIn("matched_skills", r)
            self.assertIn("matched_keywords", r)
            self.assertIsInstance(r["matched_keywords"], list)

    def test_07_industry_filter(self):
        q = "Software engineer"
        target_ind = "Software Development"
        results = self.recommender.recommend(q, top_n=5, industry_filter=target_ind)
        for r in results:
            self.assertEqual(r["industry_name"], target_ind)

    def test_08_utils(self):
        ind, exp, wt = get_filter_options(self.recommender.talent_profiles)
        self.assertGreater(len(ind), 1)
        self.assertIn("All Industries", ind)
        summary = get_dataset_summary(self.recommender.talent_profiles)
        self.assertGreater(summary["total_profiles"], 100000)

if __name__ == '__main__':
    unittest.main()
