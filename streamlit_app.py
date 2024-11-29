import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Download the 'stopwords' resource
nltk.download('stopwords')

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure page
st.set_page_config(page_title="ATS Resume Expert", page_icon="📄", layout="wide")

# Custom CSS styles for enhanced UI
st.markdown("""
    <style>
        /* Customize header */
        .css-18e3th9 {
            font-size: 28px;
            color: #4B9CD3;
            font-weight: bold;
        }
        /* Button styles */
        .stButton>button {
            background-color: #4B9CD3;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            margin: 10px 0;
            transition: transform 0.2s, background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #375A7F;
            transform: scale(1.05);
        }
        /* Style for file uploader */
        .css-1d391kg {
            border: 2px dashed #4B9CD3;
            padding: 15px;
            border-radius: 10px;
        }
        /* Response box style */
        .response-box {
            background-color: #f8f9fa;
            color: black;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #4B9CD3;
            margin-top: 20px;
            font-family: 'Arial', sans-serif;
        }
        /* Highlight keywords */
        .keyword {
            color: #4B9CD3;
            font-weight: bold;
        }
        /* Spinner style */
        .stSpinner {
            color: #4B9CD3;
        }
        /* Table Styling */
        .keyword-table {
            border-collapse: collapse;
            width: 100%;
        }
        .keyword-table th, .keyword-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .keyword-table th {
            background-color: #4B9CD3;
            color: white;
        }
        .keyword-table td {
            text-align: center;
        }
        .keyword-table .high-match {
            background-color: #d4edda;
        }
        .keyword-table .low-match {
            background-color: #f8d7da;
        }
    </style>
""", unsafe_allow_html=True)

# Gemini Response function
def get_gemini_response(input, pdf_text, job_description):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = input + "\n\nResume Content:\n" + pdf_text + "\n\nJob Description:\n" + job_description
    response = model.generate_content(prompt)
    return response.text

# Function to read PDF content
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:  # Iterate through each page of PDF
        text += page.extract_text()  # Extract text from each page
    return text

# Function to perform keyword density analysis
def keyword_density(pdf_text, job_description):
    # Initialize NLTK tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Clean and extract keywords from the job description
    job_keywords = re.findall(r'\w+', job_description.lower())
    job_keywords = [lemmatizer.lemmatize(word) for word in job_keywords if word not in stop_words]

    # Clean resume content
    resume_words = re.findall(r'\w+', pdf_text.lower())
    resume_words = [lemmatizer.lemmatize(word) for word in resume_words if word not in stop_words]

    # Count the frequency of keywords in the job description and resume
    job_keyword_count = Counter(job_keywords)
    resume_word_count = Counter(resume_words)

    # Calculate keyword density
    density = []
    for keyword, count in job_keyword_count.items():
        resume_count = resume_word_count.get(keyword, 0)
        percentage = (resume_count / count * 100) if count else 0
        match_class = 'high-match' if percentage >= 50 else 'low-match'  # Highlight high match percentage
        density.append({
            'keyword': keyword,
            'job_count': count,
            'resume_count': resume_count,
            'match_percentage': percentage,
            'match_class': match_class
        })
    
    return density

# Function to display visual keyword density chart
def plot_keyword_density_chart(density_data):
    keywords = [item['keyword'] for item in density_data]
    job_counts = [item['job_count'] for item in density_data]
    resume_counts = [item['resume_count'] for item in density_data]
    match_percentages = [item['match_percentage'] for item in density_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(keywords, match_percentages, color='skyblue')
    ax.set_xlabel('Match Percentage')
    ax.set_ylabel('Keywords')
    ax.set_title('Keyword Density Match with Job Description')
    st.pyplot(fig)

# Streamlit App Header
st.header("📄 ATS Resume Expert")

# Job Description and File Uploader Columns
input_text = st.text_area(
    "📝 Paste Job Description Here:", 
    placeholder="Insert the job description to match your resume against...",
    key="input", 
    height=150
)
uploaded_file = st.file_uploader("📂 Upload your resume (PDF format only):", type=["pdf"])

# Confirm PDF upload status
if uploaded_file:
    st.success("🎉 PDF Uploaded Successfully!", icon="✅")

# Enhanced Prompts for each button
prompts = {
    "📄 Detailed Resume Analysis": """
        You are a seasoned HR professional specializing in resume evaluation. Analyze the resume and compare it with the job description. 
        Provide detailed insights into strengths, weaknesses, and overall alignment with the job. Include recommendations for improvement.
    """,
    "💡 Skill Enhancement Suggestions": """
        You are a career growth expert. Evaluate the resume and suggest actionable ways to enhance the candidate's profile. 
        Focus on technical, soft skills, certifications, or relevant experiences that can make the resume more competitive.
    """,
    "🔍 Missing Keywords Analysis": """
        As an ATS optimization expert, compare the resume with the job description and identify critical missing keywords. 
        Suggest specific words or phrases to include for better ATS ranking, and explain their relevance to the role.
    """,
    "📊 Resume-Job Match Score": """
        As an ATS analyzer, evaluate the resume against the job description and provide a percentage match score. 
        Highlight key strengths, missing keywords, and overall suitability. Include a summary with actionable insights for better alignment.
    """
}

# Dictionary to store generated responses
generated_response = {}

# Display buttons and process responses dynamically
with st.expander("🔎 Resume Analysis Tools", expanded=True):
    for button_label, prompt_text in prompts.items():
        if st.button(button_label):
            if uploaded_file:
                pdf_content = input_pdf_text(uploaded_file)
                with st.spinner(f"✨ Generating {button_label}..."):
                    time.sleep(2)  # Simulate processing time
                    response = get_gemini_response(prompt_text, pdf_content, input_text)
                generated_response[button_label] = response
                st.markdown(f"<div class='response-box'><strong>{button_label}:</strong><br>{response}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please upload your resume to continue.", icon="⚠️")

# Visual Analytics Section for Keyword Density
with st.expander("📊 Visual Analytics", expanded=True):
    if uploaded_file and input_text:
        pdf_content = input_pdf_text(uploaded_file)
        density_data = keyword_density(pdf_content, input_text)
        plot_keyword_density_chart(density_data)

# Sidebar for extra features
st.sidebar.title("🌟 Extra Features")
st.sidebar.image(
    "https://u-static.haozhaopian.net/uid_1f26e52fc45044fbb67feeacc48a1565/aiImage/8e99f9e2b50243dc99a10a0dbccf59a2.jpg", 
    caption="Optimize your resume for success!",
    use_container_width=True
)

st.sidebar.info("""
    🚀 **Optimize Your Resume Like a Pro!**  
    Unlock powerful features to give your resume the edge it needs to shine in front of hiring managers and ATS systems.

    1. **🔍 Visual Analytics**: Dive deep into your resume's performance with detailed **keyword density** analysis. See how well your resume matches the job description, and discover which keywords need a little more attention for a better ATS score.
    2. **⚡ Skill Enhancement Suggestions**: Get insights on how to improve your skills to align better with job requirements and increase your chances of landing your dream job.
    3. **📈 Job Match Score**: Receive a percentage score on how closely your resume aligns with the job description, with suggestions for improvement.
""")
