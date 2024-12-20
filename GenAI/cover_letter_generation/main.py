import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
from chromadb import Client

## Streamlit app
st.title("Cover Letter Generator.")
st.text("Craft a highly personalized and compelling cover letter tailored to your target job, as the quality of your pitch is critically important.")
user_name = st.sidebar.text_input("Enter your name here : ")
GOOGLE_API_KEY = st.sidebar.text_input("Google API Key : ")

jd = st.text_area("Paste the Job Description here :")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please Upload Your Resume") 
submit = st.button("Submit")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def create_collection():
    client = Client()
    try : 
        collection = client.create_collection(name="cover_letter_db")
        print("Collection created successfully!!!")
        return collection
    except ValueError as e:
        print(f"An error occurred = : {e}")


def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text


def read_pdf_data(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += str(page.extract_text())
    return text    


def ingest_into_vectordb(collection, text):
    collection.add(documents=[text], metadata=[{"type": "resume", "name": user_name}], ids=["dj"])


def retrieve_rag_context(collection):
    cover_letter_prompt = f"Write a cover letter based on the profile. Consider this job description {jd}. Please fill in all my personal details from resume."
    results = collection.query(
        query_texts = [cover_letter_prompt],
        n_results = 1
    )
    context = results.get('documents')
    return context, cover_letter_prompt


def input_rag_prompt(context, cover_letter_prompt):
    # Write a prompt for RAG.
    input_prompt = f"""
    ### context"
    {context}

    ### Task : 
    using above context, answer the following question:
    {cover_letter_prompt}
    """
    return input_prompt

if submit and uploaded_file is not None:
    collection = create_collection()
    text = read_pdf_data()
    ingest_into_vectordb(collection, text)
    context, cover_letter_prompt = retrieve_rag_context(collection)
    input_prompt = input_rag_prompt(context, cover_letter_prompt)
    response = get_gemini_response(input_propmt)
    st.text(response)
