import os
import streamlit as st
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import random
import string
from dotenv import load_dotenv
load_dotenv()


def set_api_keys():
    # Setting environment variables for API keys
    OPENAIAPIKEY = st.secrets["OPENAI_API_KEY"]
    LLAMACLOUDAPIKEY = st.secrets["LLAMA_CLOUD_API_KEY"]
    return OPENAIAPIKEY

def parse_pdf_to_markdown(filepath, output_path):
    # Parsing PDF to markdown using LlamaParse
    parser = LlamaParse(result_type="markdown", num_workers=4, verbose=True, language="en")
    documents = parser.load_data(filepath)
    
    # Save the parsed markdown to a file
    with open(output_path, 'w', encoding='utf-8') as file:
        for doc in documents:
            if doc.text.strip():
                file.write(doc.text + "\n\n")
    return output_path

def split_markdown_into_chunks(md_document_path):
    # Splitting the markdown document into chunks by headers
    with open(md_document_path, 'r', encoding='utf-8') as file:
        md_document_content = file.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_chunks = markdown_splitter.split_text(md_document_content)
    
    # Log the chunks for debugging
    st.write("Markdown Header Chunks:", md_header_chunks)  # For debugging
    return md_header_chunks

def create_faiss_retriever(md_header_chunks, openai_api_key):
    # Ensure md_header_chunks is not empty
    if not md_header_chunks:
        raise ValueError("No markdown header chunks were created. Check the PDF parsing step.")
    
    # Create OpenAI embeddings and FAISS vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Ensure embeddings can be generated from md_header_chunks
    if not md_header_chunks:
        raise ValueError("No documents to create embeddings from.")
    
    try:
        vectorstore = FAISS.from_documents(md_header_chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None
    
    return vectorstore.as_retriever()

def main():
    # Set up the API keys
    openai_api_key = set_api_keys()

    # Upload a PDF file via Streamlit
    st.markdown("<h1 style = 'color:Tomato; text-align: center;'>PDF CHAT BOT </h1>", unsafe_allow_html=True)
    st.markdown("<h6>Ask questions directly to your PDF instead of searching through it for hours</h6>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_pdf:
        # Define a local directory to save the markdown file
        local_dir = os.path.expanduser("~/Documents/pdf_to_md_files")
        os.makedirs(local_dir, exist_ok=True)

        # Save the uploaded PDF file to a local system path
        pdf_filepath = os.path.join(local_dir, uploaded_pdf.name)
        with open(pdf_filepath, "wb") as f:
            f.write(uploaded_pdf.read())

        # Generate a corresponding markdown file name based on the PDF file name
        markdown_output_path = os.path.join(local_dir, f"{os.path.splitext(uploaded_pdf.name)[0]}.md")

        # Check if the Markdown file already exists
        if os.path.exists(markdown_output_path):
            st.write(f"Markdown file already exists at: {markdown_output_path}. Using the existing file.")
        else:
            st.write("Parsing PDF and generating Markdown file...")
            # Parse the PDF to Markdown and save to a file
            parse_pdf_to_markdown(pdf_filepath, markdown_output_path)
            st.write(f"Markdown file saved at: {markdown_output_path}")

        # Split the markdown document into chunks
        md_header_chunks = split_markdown_into_chunks(markdown_output_path)

        # Check if md_header_chunks is empty after parsing
        if not md_header_chunks:
            st.error("No markdown header chunks generated. Please check the PDF content.")
            return

        # Create FAISS retriever from markdown chunks
        retriever = create_faiss_retriever(md_header_chunks, openai_api_key)

        # Create RAG chain for question answering
        rag_chain = create_rag_chain(retriever, openai_api_key)

        # Ask multiple questions and store question-answer pairs
        ask_questions(rag_chain)


if __name__ == "__main__":
    main()
