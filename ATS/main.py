import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text


def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += str(page.extract_text())
    return text    


## Prompt Template 
input_propmt = """
Hey Act like a skilled or very experience ATS (Application Tracking System) with deep understanding of tech field, software engineering, data science, machine learning engineer, data analyst and big data engineer. Your task is to evaluate the resume based on the job description. You must consider the job market is very competitive and you should provide best assistance for improving the resume. Assign the percentage maching score based on job description and missing keywords with high accuracy.
resume : {text}
description : {jd}

I want the response in one single string having the structure 
{{"JD Match" : "%", \n "Missing Keywords : []", \n "Profile Summary" : ""}}
"""

## Streamlit app
st.title("Smart ATS")
st.text("Improve Your Resume ATS")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please Upload Your Resume") 

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        response = get_gemini_response(input_propmt)
        st.subheader(response)
