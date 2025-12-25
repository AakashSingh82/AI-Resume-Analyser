import streamlit as st
import re
import pandas as pd
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ================= CONFIG =================
st.set_page_config(
    page_title="AI Resume Analyzer & Recruitment Agent",
    page_icon="🤖",
    layout="wide"
)

# ================= FUNCTIONS =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = Document(file)
    return " ".join(p.text for p in doc.paragraphs)

def calculate_ats(resume_text):
    benchmark = """
    skills experience education projects certifications achievements
    python sql machine learning data analysis communication leadership
    problem solving teamwork impact results
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, benchmark])
    return cosine_similarity(vectors)[0][1] * 100

def generate_pdf(report_text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    content = [Paragraph(line, styles["Normal"]) for line in report_text.split("\n")]
    doc.build(content)
    return tmp.name

# ================= SIDEBAR =================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose Module",
    ["🧠 AI Resume Analyzer (Employee)", "🏢 Recruitment Agent (Employer)"]
)

# ==================================================
# =============== EMPLOYEE ==========================
# ==================================================
if page == "🧠 AI Resume Analyzer (Employee)":
    st.title("🧠 AI Resume Analyzer")
    st.write("Detailed ATS + recruiter-level resume evaluation")

    uploaded_resume = st.file_uploader(
        "📂 Upload Resume (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_resume and st.button("Run Detailed Analysis", use_container_width=True):

        if uploaded_resume.name.endswith(".pdf"):
            resume_text = read_pdf(uploaded_resume)
        elif uploaded_resume.name.endswith(".docx"):
            resume_text = read_docx(uploaded_resume)
        else:
            resume_text = uploaded_resume.read().decode("utf-8")

        clean_resume = clean_text(resume_text)
        ats = calculate_ats(clean_resume)

        # Selection Probability (realistic scaling)
        selection_prob = min(95, round((ats * 0.85) + 10, 1))

        # ================= METRICS =================
        st.subheader("📊 Resume Evaluation Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("ATS Score", f"{ats:.2f}%")
        c2.metric("Selection Probability", f"{selection_prob}%")
        c3.metric("Recruiter Verdict",
                  "Strong" if ats > 75 else "Average" if ats > 50 else "Weak")

        # ================= ATS BREAKDOWN =================
        st.subheader("📈 ATS Score Breakdown")
        breakdown = {
            "Keyword Match": ats,
            "Formatting (ATS Safe)": 85 if ats > 60 else 60,
            "Section Coverage": 90 if "skills" in clean_resume and "experience" in clean_resume else 50,
            "Readability": 80 if 400 <= len(resume_text.split()) <= 900 else 55
        }

        fig, ax = plt.subplots()
        ax.bar(breakdown.keys(), breakdown.values())
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        st.pyplot(fig)

        # ================= DEEP ANALYSIS =================
        st.subheader("🔍 Deep Resume Analysis")

        # Section Analysis
        sections = ["skills", "experience", "projects", "education", "certifications"]
        for sec in sections:
            if sec in clean_resume:
                st.success(f"✔ {sec.title()} section found")
            else:
                st.error(f"✘ {sec.title()} section missing")

        # Keyword Density Check
        words = clean_resume.split()
        keyword_density = len(set(words)) / max(len(words), 1)

        st.write(f"• Keyword diversity ratio: **{keyword_density:.2f}**")
        if keyword_density < 0.35:
            st.warning("Low keyword diversity → ATS may consider content repetitive")

        # Recruiter Scan Insight
        st.subheader("👀 Recruiter 6–8 Second Scan Result")
        if ats < 50:
            st.error("Likely rejected in first scan due to weak keyword alignment")
        elif ats < 75:
            st.warning("May pass ATS but recruiter interest is average")
        else:
            st.success("Strong first impression for recruiter")

        # ================= IMPROVEMENT GUIDE =================
        st.subheader("🚀 How to Improve Resume for Higher ATS & Selection")

        st.markdown("""
        ### 🔹 1. Bullet Point Formula (VERY IMPORTANT)
        **Action Verb + What You Did + Tool/Skill + Result**
        - ❌ Worked on data analysis  
        - ✅ Analyzed sales data using Python, improving forecast accuracy by 18%

        ### 🔹 2. Skills Optimization
        - Add **8–12 job-relevant skills**
        - Avoid generic skills like *hardworking*
        - Match skills exactly as written in job descriptions

        ### 🔹 3. Keyword Placement Strategy
        - Skills section (primary)
        - Experience bullets (secondary)
        - Projects section (contextual)
        - Repeat key skills **2–3 times max**

        ### 🔹 4. Experience Section (Most Important)
        - 3–5 bullets per role
        - Start every bullet with a strong action verb
        - Quantify impact using numbers, %, time saved

        ### 🔹 5. Projects Section (For Freshers)
        - Problem → Approach → Tools → Outcome
        - Mention datasets, APIs, or real-world use

        ### 🔹 6. ATS-Safe Formatting Rules
        - No tables, text boxes, icons, images
        - Use Arial / Calibri
        - Black & white only
        - Save as PDF

        ### 🔹 7. Tailoring Strategy (BOOSTS SELECTION MASSIVELY)
        - Modify resume keywords for every job role
        - Never send the same resume everywhere
        """)

        # ================= FINAL CHECKLIST =================
        st.subheader("✅ High-Selection Resume Checklist")
        st.markdown("""
        ✔ Clear section headings  
        ✔ Quantified achievements  
        ✔ Job-specific keywords  
        ✔ Clean formatting  
        ✔ 1–2 page length  
        ✔ PDF format  
        """)

        # ================= PDF REPORT =================
        report = f"""
        AI RESUME FEEDBACK REPORT

        ATS Score: {ats:.2f}%
        Selection Probability: {selection_prob}%

        Key Recommendations:
        - Improve keyword alignment
        - Quantify experience
        - Optimize bullet structure
        - Follow ATS-safe formatting

        Final Insight:
        Focus on impact-driven bullets and keyword optimization.
        """

        pdf_path = generate_pdf(report)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download AI Feedback PDF",
                f,
                file_name="AI_Resume_Feedback.pdf",
                mime="application/pdf"
            )

# ==================================================
# =============== EMPLOYER ==========================
# ==================================================
if page == "🏢 Recruitment Agent (Employer)":
    st.title("🏢 Recruitment Agent")
    st.write("Automated resume screening and candidate shortlisting")

    job_desc = st.text_area("🧾 Enter Job Description", height=220)
    uploaded_files = st.file_uploader(
        "📂 Upload Candidate Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Run Automated Screening", use_container_width=True):
        if uploaded_files and job_desc.strip():
            job_clean = clean_text(job_desc)
            results = []

            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = read_pdf(file)
                elif file.name.endswith(".docx"):
                    text = read_docx(file)
                else:
                    text = file.read().decode("utf-8")

                resume_clean = clean_text(text)
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([resume_clean, job_clean])
                score = cosine_similarity(vectors)[0][1] * 100

                results.append({
                    "Candidate": file.name,
                    "Match Score (%)": round(score, 2),
                    "Decision": "Shortlisted" if score >= 70 else "Rejected"
                })

            df = pd.DataFrame(results).sort_values("Match Score (%)", ascending=False)
            st.dataframe(df, use_container_width=True)
