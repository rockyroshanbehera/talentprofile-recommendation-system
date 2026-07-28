import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="TalentMatch AI",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_resources():

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    talent_profiles = joblib.load("models/talent_profiles.pkl")

    # Recreate TF-IDF Matrix
    tfidf_matrix = vectorizer.transform(
        talent_profiles["combined_text"]
    )

    return vectorizer, talent_profiles, tfidf_matrix


vectorizer, talent_profiles, tfidf_matrix = load_resources()

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(query, top_n=10):

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity.argsort()[-top_n:][::-1]

    return talent_profiles.iloc[top_indices].copy(), similarity[top_indices]

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("TalentMatch AI")

st.sidebar.info(
"""
NLP Based Talent Recommendation System

✔ TF-IDF
✔ Cosine Similarity
✔ Streamlit
✔ Machine Learning
"""
)

# -------------------------------
# Header
# -------------------------------
st.title("🎯 TalentMatch AI")

st.markdown(
"""
### NLP-Based Talent Recommendation System

Search talent using natural language.

Example:

- Python Data Scientist
- Machine Learning Engineer
- Data Analyst SQL
- Frontend React Developer
"""
)

query = st.text_input(
    "Search Talent",
    placeholder="Example: Python Data Scientist"
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# -------------------------------
# Recommendation Button
# -------------------------------
if st.button("Recommend Talent"):

    if query.strip() == "":
        st.warning("Please enter a search query.")
    else:

        results, scores = recommend(query, top_n)

        st.success(f"Top {top_n} Matching Profiles")

        for i, (_, row) in enumerate(results.iterrows()):

            score = scores[i]

            with st.container():

                st.markdown("---")

                st.subheader(f"#{i+1}")

                # Job Title
                if "title" in row.index:
                    st.write("**Job Title:**", row["title"])

                # Company
                if "company_name" in row.index:
                    st.write("**Company:**", row["company_name"])

                # Location
                if "location" in row.index:
                    st.write("**Location:**", row["location"])

                # Skills
                if "skills_desc" in row.index:
                    st.write("**Skills:**")
                    st.write(row["skills_desc"])

                # Description
                if "description" in row.index:
                    with st.expander("Job Description"):
                        st.write(row["description"])

                st.progress(float(score))

                st.write(f"Similarity Score: {score:.2f}")
