"""
AI Job Application Agent
Streamlit + LangGraph Multi-Agent Application
"""

import os
import tempfile
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()


def get_groq_api_key():
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


groq_api_key = get_groq_api_key()


# ==============================
# PROFESSIONAL CSS
# ==============================

st.markdown("""
<style>
:root {
    --bg: #07111f;
    --panel: rgba(15, 23, 42, 0.88);
    --card: #ffffff;
    --line: #dbeafe;
    --text: #0f172a;
    --muted: #475569;
    --primary: #2563eb;
    --accent: #8b5cf6;
    --success: #0f766e;
    --soft: #eff6ff;
}

html, body, .stApp {
    background:
        radial-gradient(circle at top, #eff6ff 0%, #f8fbff 45%, #eef2ff 100%);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.8rem;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
    max-width: 1320px;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #111827 0%, #172554 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

.sidebar-card, .mini-card, .glass-card, .info-card, .input-card, .step-card {
    border-radius: 22px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.sidebar-card:hover, .mini-card:hover, .glass-card:hover, .info-card:hover, .input-card:hover, .step-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 28px rgba(37, 99, 235, 0.12);
}

.sidebar-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    padding: 12px 14px;
    margin-bottom: 10px;
}

.api-success {
    background: linear-gradient(135deg, #064E3B, #047857);
    color: #ECFDF5 !important;
    padding: 12px 14px;
    border-radius: 14px;
    font-weight: 700;
    border: 1px solid rgba(167, 243, 208, 0.25);
}

.hero-card {
    background:
        linear-gradient(135deg, #0F172A 0%, #1E3A8A 45%, #7C3AED 100%);
    border: 1px solid rgba(219, 234, 254, 0.18);
    border-radius: 30px;
    padding: 30px 32px;
    margin-bottom: 18px;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.18);
}

.hero-title {
    font-size: 42px;
    line-height: 1.06;
    font-weight: 900;
    color: #F8FAFC;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 17px;
    line-height: 1.6;
    color: #E0F2FE;
    max-width: 980px;
}

.feature-row {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.feature-pill {
    background: rgba(248, 250, 252, 0.12);
    color: #F8FAFC;
    border: 1px solid rgba(191, 219, 254, 0.35);
    padding: 8px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 16px;
}

.metric-card {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 12px 14px;
    color: #EFF6FF;
}

.metric-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #BFDBFE;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 20px;
    font-weight: 800;
}

.info-card, .glass-card, .step-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
    border: 1px solid #E5E7EB;
    border-radius: 24px;
    padding: 16px;
    margin-bottom: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 6px;
}

.help-text, .subtle-text {
    color: #64748B;
    font-size: 14px;
    line-height: 1.55;
}

.tip-box {
    #  background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
    border: 1px solid #C4B5FD;
    border-radius: 18px;
    padding: 12px 13px;
    color: #312E81;
    font-size: 14px;
}

.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}

.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    # background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
    border-radius: 999px;
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #93C5FD !important;
    border-radius: 18px !important;
    padding: 14px !important;
}

[data-testid="stFileUploader"] label {
    color: #0F172A !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%) !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 16px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 18px rgba(37, 99, 235, 0.18);
}

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: #0F172A !important;
    font-weight: 800 !important;
}

.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 10px 12px !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12) !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #94A3B8 !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.warning-box {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    color: #92400E;
    border: 1px solid #FCD34D;
    border-radius: 18px;
    padding: 14px 16px;
    font-weight: 700;
    margin-top: 12px;
}

.success-box {
    background: linear-gradient(135deg, #ECFDF5, #F0FDF4);
    color: #065F46;
    border: 1px solid #86EFAC;
    border-radius: 18px;
    padding: 16px 18px;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    box-shadow: 0 12px 22px rgba(37, 99, 235, 0.18);
    transition: transform 140ms ease, box-shadow 140ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 24px rgba(124, 58, 237, 0.22);
}

button[data-baseweb="tab"] {
    font-weight: 800 !important;
    color: #334155 !important;
    border-radius: 12px 12px 0 0 !important;
}

button[data-baseweb="tab"]:hover {
    color: #111827 !important;
    background: #F8FAFC !important;
}

.stMarkdown, .stText, p, li {
    color: #0F172A;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    border: none !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(124, 58, 237, 0.22) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #6D28D9) !important;
    color: #FFFFFF !important;
    box-shadow: 0 14px 24px rgba(37, 99, 235, 0.25) !important;

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 30px; }
    .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
}
</style>
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">💼 AI Job Application Agent</div>
    <div class="hero-subtitle">
        Upload your resume, paste a job URL, and let a guided AI workflow generate a polished job application package in minutes.
        The experience is simple, visual, and friendly for beginners and professionals alike.
    </div>
    <div class="feature-row">
        <div class="feature-pill">🤖 Multi-Agent Workflow</div>
        <div class="feature-pill">📄 Resume + JD Analysis</div>
        <div class="feature-pill">✍️ Tailored Cover Letter</div>
        <div class="feature-pill">📊 Match Score + Skill Gaps</div>
        <div class="feature-pill">⬇️ One-click Downloads</div>
    </div>
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-title">Fast Setup</div><div class="metric-value">3 simple steps</div></div>
        <div class="metric-card"><div class="metric-title">AI Assist</div><div class="metric-value">6 smart agents</div></div>
        <div class="metric-card"><div class="metric-title">Outcome</div><div class="metric-value">Ready to apply</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    st.markdown("## ✨ How this works")
    st.markdown("""
    <div class="tip-box">
      1. Upload your resume<br>
      2. Add a job URL or paste the JD<br>
      3. Generate your tailored package
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌟 Why users love it")
    st.info("Clear steps, readable outputs, and helpful suggestions for beginners and professionals alike.")

    st.markdown("""
    <div class="badge-row">
      <span class="badge-chip">🧠 AI-guided</span>
      <span class="badge-chip">📱 Mobile friendly</span>
      <span class="badge-chip">⚡ Fast workflow</span>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        "JD Extractor Agent",
        "Resume Analyzer Agent",
        "Match Score Agent",
        "Cover Letter Agent",
        "Skill Gap Agent",
        "Final Package Agent",
    ]

    for i, step in enumerate(steps, 1):
        st.markdown(
            f"""
            <div class="sidebar-card">
                <b>{i}. {step}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if groq_api_key:
        st.markdown('<div class="api-success">✅ GROQ API Key Loaded</div>', unsafe_allow_html=True)
    else:
        st.error("GROQ API Key Missing")

    # st.markdown("""
    # ### Deployment Note
    # Locally, use `.env`.

    # On Streamlit Cloud, add `GROQ_API_KEY` in **Secrets**.
    # """)

    st.markdown("---")
    st.caption("Tip: If the URL does not load, use the manual job description box below.")


if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it in `.env` locally or Streamlit Cloud Secrets.")
    st.stop()


# ==============================
# LLM
# ==============================

@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=api_key,
    )


llm = load_llm(groq_api_key)


# ==============================
# STATE
# ==============================

class JobApplicationState(TypedDict):
    resume_path: str
    jd_url: str
    jd_manual_text: str
    jd_raw_text: str
    job_details: str
    resume_raw_text: str
    resume_summary: str
    match_analysis: str
    cover_letter: str
    skill_gap_analysis: str
    final_package: str


# ==============================
# HELPERS
# ==============================

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ["http", "https"] and bool(parsed.netloc)
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def scrape_job_description(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) < 200:
            return ""

        return clean_text[:10000]

    except Exception:
        return ""


def read_resume_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "\n\n".join(doc.page_content for doc in documents)
    return text[:10000]


# ==============================
# AGENTS
# ==============================

def jd_extractor_agent(state: JobApplicationState):
    jd_raw_text = ""

    if state.get("jd_url") and is_valid_url(state["jd_url"]):
        jd_raw_text = scrape_job_description(state["jd_url"])

    if not jd_raw_text and state.get("jd_manual_text"):
        jd_raw_text = state["jd_manual_text"][:10000]

    if not jd_raw_text:
        jd_raw_text = "No job description text could be extracted."

    prompt = ChatPromptTemplate.from_template("""
You are a Job Description Extractor Agent.

Read the raw job posting text and extract structured job details.
Do not invent missing details. If a detail is not available, write "Not mentioned".

Raw Job Description:
{jd_raw_text}

Return output in this format:

Company Name:
Job Role:
Location:
Experience Required:
Required Skills:
Responsibilities:
Preferred Qualifications:
Important Keywords:
""")

    response = llm.invoke(prompt.format_messages(jd_raw_text=jd_raw_text))

    return {
        "jd_raw_text": jd_raw_text,
        "job_details": response.content,
    }


def resume_analyzer_agent(state: JobApplicationState):
    resume_raw_text = read_resume_pdf(state["resume_path"])

    prompt = ChatPromptTemplate.from_template("""
You are a Resume Analyzer Agent.

Analyze the resume text and extract the candidate profile.
Do not invent missing details. If a detail is not available, write "Not mentioned".

Resume Text:
{resume_raw_text}

Return output in this format:

Candidate Name:
Professional Summary:
Education:
Technical Skills:
Projects:
Work Experience:
Certifications:
Achievements:
Strengths:
""")

    response = llm.invoke(prompt.format_messages(resume_raw_text=resume_raw_text))

    return {
        "resume_raw_text": resume_raw_text,
        "resume_summary": response.content,
    }


def match_score_agent(state: JobApplicationState):
    prompt = ChatPromptTemplate.from_template("""
You are a Job Match Scoring Agent.

Compare the candidate resume summary with the job details.
Be realistic and honest. Do not overrate the candidate.

Job Details:
{job_details}

Resume Summary:
{resume_summary}

Return output in this format:

Match Score out of 10:
Match Percentage:
Strong Matching Skills:
Partially Matching Skills:
Missing or Weak Skills:
Reason for Score:
Hiring Recommendation:
""")

    response = llm.invoke(
        prompt.format_messages(
            job_details=state["job_details"],
            resume_summary=state["resume_summary"],
        )
    )

    return {"match_analysis": response.content}


def cover_letter_agent(state: JobApplicationState):
    prompt = ChatPromptTemplate.from_template("""
You are a Professional Cover Letter Writer Agent.

Write a tailored cover letter using the job details, resume summary, and match analysis.

Job Details:
{job_details}

Resume Summary:
{resume_summary}

Match Analysis:
{match_analysis}

Rules:
- Professional tone
- Do not exaggerate
- Keep it under 250 words
- Highlight only relevant skills
- Make it suitable for HR
- Do not include placeholders like [Your Name] if the candidate name is available

Cover Letter:
""")

    response = llm.invoke(
        prompt.format_messages(
            job_details=state["job_details"],
            resume_summary=state["resume_summary"],
            match_analysis=state["match_analysis"],
        )
    )

    return {"cover_letter": response.content}


def skill_gap_agent(state: JobApplicationState):
    prompt = ChatPromptTemplate.from_template("""
You are a Skill Gap Analysis Agent.

Compare the job details with the resume summary and identify skill gaps.
Keep suggestions practical and beginner-friendly.

Job Details:
{job_details}

Resume Summary:
{resume_summary}

Match Analysis:
{match_analysis}

Return output in this format:

Top Missing Skills:
Why These Skills Matter:
30-Day Learning Plan:
Free Learning Resources:
Portfolio Project Suggestions:
""")

    response = llm.invoke(
        prompt.format_messages(
            job_details=state["job_details"],
            resume_summary=state["resume_summary"],
            match_analysis=state["match_analysis"],
        )
    )

    return {"skill_gap_analysis": response.content}


def final_package_agent(state: JobApplicationState):
    final_package = f"""
AI JOB APPLICATION PACKAGE

==================================================
1. JOB DETAILS
==================================================
{state["job_details"]}

==================================================
2. RESUME SUMMARY
==================================================
{state["resume_summary"]}

==================================================
3. MATCH ANALYSIS
==================================================
{state["match_analysis"]}

==================================================
4. TAILORED COVER LETTER
==================================================
{state["cover_letter"]}

==================================================
5. SKILL GAP ANALYSIS
==================================================
{state["skill_gap_analysis"]}
"""

    return {"final_package": final_package}


# ==============================
# LANGGRAPH
# ==============================

@st.cache_resource(show_spinner=False)
def build_graph():
    workflow = StateGraph(JobApplicationState)

    workflow.add_node("jd_extractor", jd_extractor_agent)
    workflow.add_node("resume_analyzer", resume_analyzer_agent)
    workflow.add_node("match_scorer", match_score_agent)
    workflow.add_node("cover_letter_writer", cover_letter_agent)
    workflow.add_node("skill_gap_analyzer", skill_gap_agent)
    workflow.add_node("final_packager", final_package_agent)

    workflow.add_edge(START, "jd_extractor")
    workflow.add_edge("jd_extractor", "resume_analyzer")
    workflow.add_edge("resume_analyzer", "match_scorer")
    workflow.add_edge("match_scorer", "cover_letter_writer")
    workflow.add_edge("cover_letter_writer", "skill_gap_analyzer")
    workflow.add_edge("skill_gap_analyzer", "final_packager")
    workflow.add_edge("final_packager", END)

    return workflow.compile()


app = build_graph()


# ==============================
# INPUT UI
# ==============================

st.markdown("## Start Your Application Package")

st.markdown("""
<div class="info-card">
  <div class="section-title">🎯 Simple workflow, polished results</div>
  <div class="help-text">Choose your resume, add the job description, and generate a polished application package in one click. Each section is clearly organized so you can review, refine, and download with confidence.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
  <span class="badge-chip">✅ Resume analysis</span>
  <span class="badge-chip">📊 Match scoring</span>
  <span class="badge-chip">📝 Cover letter drafting</span>
  <span class="badge-chip">📚 Skill-gap guidance</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Resume Upload</div>', unsafe_allow_html=True)
    uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])
    st.markdown('<div class="help-text">Upload a clean PDF resume. Text-based PDF works best and improves extraction quality.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔗 Job Description URL</div>', unsafe_allow_html=True)
    jd_url = st.text_input(
        "Paste Job URL",
        placeholder="https://example.com/job-posting",
    )
    st.markdown('<div class="help-text">Use a public job page URL. If the site blocks scraping, use the manual job description box below.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📝 Optional: Paste Job Description manually if URL is blocked"):
    jd_manual_text = st.text_area(
        "Paste JD text here",
        height=190,
        placeholder="Paste the full job description here if URL extraction does not work.",
    )

st.markdown("""
<div class="warning-box">
⚠️ Some job websites block scraping or load content using JavaScript. If the URL does not work, paste the job description manually in the optional box.
</div>
""", unsafe_allow_html=True)

st.write("")
generate_button = st.button("🚀 Generate Application Package", use_container_width=True)


# ==============================
# RUN WORKFLOW
# ==============================

if generate_button:
    if not uploaded_resume:
        st.error("Please upload your resume PDF.")
        st.stop()

    if not jd_url.strip() and not jd_manual_text.strip():
        st.error("Please paste a job description URL or manually paste the job description text.")
        st.stop()

    if jd_url.strip() and not is_valid_url(jd_url.strip()):
        st.error("Please enter a valid URL starting with http:// or https://, or use the manual JD box.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_resume.read())
        resume_path = temp_file.name

    initial_state = {
        "resume_path": resume_path,
        "jd_url": jd_url.strip(),
        "jd_manual_text": jd_manual_text.strip(),
        "jd_raw_text": "",
        "job_details": "",
        "resume_raw_text": "",
        "resume_summary": "",
        "match_analysis": "",
        "cover_letter": "",
        "skill_gap_analysis": "",
        "final_package": "",
    }

    progress = st.progress(0)
    status = st.empty()
    final_state = initial_state.copy()

    steps = {
        "jd_extractor": "Agent 1: Extracting job description...",
        "resume_analyzer": "Agent 2: Reading and analyzing resume...",
        "match_scorer": "Agent 3: Calculating job match score...",
        "cover_letter_writer": "Agent 4: Writing tailored cover letter...",
        "skill_gap_analyzer": "Agent 5: Finding skill gaps...",
        "final_packager": "Agent 6: Preparing final package...",
    }

    progress_count = 0
    total_steps = len(steps)

    try:
        with st.spinner("Multi-agent workflow is running..."):
            for step in app.stream(initial_state):
                agent_name = list(step.keys())[0]
                progress_count += 1
                progress.progress(min(progress_count / total_steps, 1.0))

                if agent_name in steps:
                    status.info(steps[agent_name])

                final_state.update(step[agent_name])

        progress.progress(1.0)
        status.success("All agents completed successfully!")
        st.session_state.final_state = final_state

    except Exception as e:
        st.error(f"Workflow failed: {e}")

    finally:
        try:
            os.remove(resume_path)
        except Exception:
            pass


# ==============================
# DISPLAY RESULTS
# ==============================

if "final_state" in st.session_state:
    final_state = st.session_state.final_state

    st.markdown("""
<div class="success-box">
✅ Your AI Job Application Package is ready. Review the outputs below and download what you need.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Match Score", "Ready", "AI generated")
    c2.metric("Cover Letter", "Personalized", "HR friendly")
    c3.metric("Skill Gap", "Actionable", "Learning tips")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Job Details",
        "👤 Resume Summary",
        "📊 Match Analysis",
        "✍️ Cover Letter",
        "📚 Skill Gaps",
        "📦 Full Package",
    ])

    with tab1:
        st.subheader("📌 Extracted Job Details")
        st.write(final_state["job_details"])
        with st.expander("View Raw JD Text"):
            st.write(final_state["jd_raw_text"])

    with tab2:
        st.subheader("👤 Resume Summary")
        st.write(final_state["resume_summary"])
        with st.expander("View Raw Resume Text"):
            st.write(final_state["resume_raw_text"])

    with tab3:
        st.subheader("📊 Match Analysis")
        st.write(final_state["match_analysis"])

    with tab4:
        st.subheader("✍️ Tailored Cover Letter")
        st.write(final_state["cover_letter"])
        st.download_button(
            label="⬇️ Download Cover Letter",
            data=final_state["cover_letter"],
            file_name="cover_letter.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab5:
        st.subheader("📚 Skill Gap Analysis")
        st.write(final_state["skill_gap_analysis"])

    with tab6:
        st.subheader("📦 Full Application Package")
        st.text_area(
            "Complete Package",
            value=final_state["final_package"],
            height=500,
        )
        st.download_button(
            label="⬇️ Download Full Application Package",
            data=final_state["final_package"],
            file_name="application_package.txt",
            mime="text/plain",
            use_container_width=True,
        )