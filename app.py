import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(page_title="AI Search Assistant", page_icon="🤖")

st.title("AI Search Assistant")
st.write("Web Search + PDF Study")

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

mode = st.selectbox(
    "Choose mode",
    ["Web Search", "PDF Study",]
)

question = st.text_input("Ask your question")

uploaded_file = None

if mode in ["PDF Study",]:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


def web_search(query):
    results = list(DDGS().text(query, max_results=5))

    if not results:
        return "No web results found."

    web_data = "\n\n".join(
        [
            f"Title: {r.get('title', '')}\n"
            f"Link: {r.get('href', '')}\n"
            f"Summary: {r.get('body', '')}"
            for r in results
        ]
    )

    return web_data


def pdf_rag(file, query):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    docs = vectorstore.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context


if st.button("Generate Answer"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif mode in ["PDF Study"] and uploaded_file is None:
        st.warning("Please upload a PDF.")

    else:
        web_context = ""
        pdf_context = ""

        with st.spinner("Collecting information..."):

            if mode in ["Web Search"]:
                web_context = web_search(question)

            if mode in ["PDF Study",]:
                pdf_context = pdf_rag(uploaded_file, question)

        prompt = f"""
You are a helpful AI research assistant.

Answer the question using the available context.

Question:
{question}

Web Search Context:
{web_context}

PDF Context:
{pdf_context}

Instructions:
- If PDF context is available, prioritize it.
- If web context is available, use it for additional support.
- Give a clear, detailed and structured answer.
- If you don't know the answer, say you don't know. Don't make up information.
- Provide only the detailed and required answer without additional comments or unnecesary information.
- Generate detailed answers with step-by-step reasoning if the question is complex. 
"""

        with st.spinner("Generating Summary...."):
            response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)