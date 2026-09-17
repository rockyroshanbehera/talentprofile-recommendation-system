"""
app.py - Streamlit Application for TalentProfile AI
AI-powered talent discovery from natural-language queries.
"""

import time
import streamlit as st
import pandas as pd
from src.recommendation import TalentRecommender
from src.utils import get_filter_options, get_dataset_summary

# Page configuration
st.set_page_config(
    page_title="TalentProfile AI - Universal Talent Recommendation System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern executive recruiter UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .rec-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .rec-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .badge-score {
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.88rem;
        border: 1px solid #BFDBFE;
    }
    .badge-tag {
        background-color: #F1F5F9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 6px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .badge-matched {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
        border: 1px solid #A7F3D0;
    }
    .why-matches {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="? Loading Talent Knowledge Base & Vector Models...")
def load_recommender():
    recommender = TalentRecommender()
    recommender.load_models()
    return recommender


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.sidebar.title("Recommendation Controls")

    try:
        recommender = load_recommender()
        industries, experiences, work_types = get_filter_options(recommender.talent_profiles)
        summary = get_dataset_summary(recommender.talent_profiles)
    except Exception as e:
        st.error(f"? Initialization Error: Could not load models. Detail: {str(e)}")
        st.info("Please ensure 	fidf_vectorizer.pkl, 	fidf_matrix.pkl, and 	alent_profiles.pkl are in the models/ directory.")
        return

    # Dataset stats in sidebar
    st.sidebar.caption(f"?? Indexed Knowledge Base: **{summary['total_profiles']:,}** Profiles")

    top_n = st.sidebar.select_slider("Results to return", options=[5, 10, 15, 20], value=10)

    st.sidebar.markdown("### ?? Refinement Filters")
    selected_industry = st.sidebar.selectbox("Industry", industries, index=0)
    selected_exp = st.sidebar.selectbox("Experience Level", experiences, index=0)
    selected_work_type = st.sidebar.selectbox("Work Type", work_types, index=0)
    location_keyword = st.sidebar.text_input("Location (e.g. New York, CA, London)", value="")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ?? Query Inspiration")
    example_queries = [
        "Need a Python data analyst with SQL and Power BI experience",
        "Looking for a machine learning engineer with Python, TensorFlow and NLP skills",
        "Need a marketing professional experienced in digital marketing and SEO",
        "Senior cloud DevOps architect with Kubernetes, Docker and AWS",
        "UI/UX designer with Figma, wireframing and user research experience"
    ]
    for ex in example_queries:
        if st.sidebar.button(ex, key=f"btn_{ex[:15]}"):
            st.session_state["query_input"] = ex

    # Main UI Header
    st.markdown('<div class="main-title">?? TalentProfile AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered talent discovery from natural-language recruiter requirements</div>', unsafe_allow_html=True)

    # Search Box
    default_q = st.session_state.get("query_input", "Need a Python data analyst with SQL and Power BI experience")
    
    col_input, col_btn1, col_btn2 = st.columns([5, 1, 1])
    with col_input:
        query = st.text_input(
            "Enter recruiter job description, required skills, or candidate requirements:",
            value=default_q,
            placeholder="e.g. Looking for a Python machine learning engineer with NLP and PyTorch experience...",
            label_visibility="collapsed"
        )
    with col_btn1:
        search_clicked = st.button("?? Find Talent", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("?? Reset", use_container_width=True)

    if clear_clicked:
        st.session_state["query_input"] = ""
        st.rerun()

    # Trigger Search
    if search_clicked or query:
        if not query.strip():
            st.warning("?? Please enter a recruiter query or select an example prompt to search for candidates.")
            return

        t_start = time.time()
        with st.spinner("Running TF-IDF Vectorization & Cosine Similarity Ranking..."):
            results = recommender.recommend(
                query=query,
                top_n=top_n,
                industry_filter=selected_industry,
                experience_filter=selected_exp,
                work_type_filter=selected_work_type,
                location_keyword=location_keyword
            )
        elapsed_ms = (time.time() - t_start) * 1000

        # Results header
        st.markdown(f"### ?? Ranked Recommendations ({len(results)} matches in {elapsed_ms:.1f}ms)")

        if not results:
            st.info("?? No talent profiles matched your active query and filter combination. Try loosening filters or broadening your search terms.")
            return

        for rec in results:
            with st.container():
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">#{rec['rank']} {rec['title']}</span>
                            <div style="color: #64748B; font-size: 0.95rem; margin-top: 2px;">
                                ?? <strong>{rec['company_name']}</strong> &nbsp;|&nbsp; 
                                ?? {rec['industry_name']} &nbsp;|&nbsp; 
                                ?? {rec['location']}
                            </div>
                        </div>
                        <div>
                            <span class="badge-score">Match Score: {rec['similarity_percentage']}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Why this matches section (Explainable AI)
                matched_items = []
                if rec["matched_skills"]:
                    matched_items.extend([f"? Skill: <strong>{s}</strong>" for s in rec["matched_skills"]])
                if rec["matched_keywords"]:
                    matched_items.extend([f"? Keyword: <strong>{k}</strong>" for k in rec["matched_keywords"]])

                if matched_items:
                    st.markdown(f"""
                    <div class="why-matches">
                        <strong style="color: #065F46;">?? Why this profile matches:</strong><br/>
                        {' &nbsp;&nbsp;|&nbsp;&nbsp; '.join(matched_items)}
                    </div>
                    """, unsafe_allow_html=True)

                # Skills, Experience & Meta tags
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Mapped Skills:** {rec['skill_name'] if rec['skill_name'] else 'General'}")
                    st.markdown(f"**Profile Summary:** {rec['short_description']}")
                with col2:
                    st.markdown(f"**Experience:** {rec['formatted_experience_level']}")
                    st.markdown(f"**Work Type:** {rec['formatted_work_type']}")

                # Detailed description accordion
                with st.expander("?? View Complete Candidate Experience & Job Details"):
                    st.write(rec["full_description"])

                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
