import os
import bs4
import streamlit as st
import pickle
import time
from langchain.document_loaders import UnstructuredURLLoader
#from langchain_community.document_loaders.url import UnstructuredURLLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

vectorstore_name = 'faiss_store_openai'
model = ChatOpenAI(model="gpt-3.5-turbo")
load_dotenv()

def load_and_index_data(urls: list):
    # This method loads data, split into chunks, generate embeddings and stores the embeddings.
    # Load the data coming from URLs:
    '''
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    '''
    loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
    data = loader.load()

    # Splitting the data 
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    return vectorstore_openai

def save_vector_store(vectorstore, vectorstore_name : str):
    embeddings = OpenAIEmbeddings()
    vectorstore.save_local(vectorstore_name)
    st.sidebar.write("Vector Store Saved Successfully......")


def load_vector_store(vectorstore_name:str):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vectorstore_name, embeddings, allow_dangerous_deserialization=True)
    print("Vector Store Loaded Successfully......")
    return vector_store


def streamlit_frontend():
    st.title("News Research Tool")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)
    
    process_url_clicked = st.sidebar.button("Load External Data") 

    if process_url_clicked:
        vectorstore_openai = load_and_index_data(urls)
        save_vector_store(vectorstore_openai, vectorstore_name)

    
    query = st.text_input("Question :", placeholder = "Type your question here..")
    process_go_clicked = st.button("Go")
    if process_go_clicked : 
        if os.path.exists(vectorstore_name):
            # Load vector store
            vector_store = load_vector_store(vectorstore_name)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=model, retriever=vector_store.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Output From LLM : ")
            st.write(result["answer"])

if __name__=="__main__":
    streamlit_frontend()