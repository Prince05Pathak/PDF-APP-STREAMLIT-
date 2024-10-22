import os
import streamlit as st
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from giskard import GiskardPromptGenerator  # Assuming this is the package for Giskard
from ragas import RAGASEvaluator  # Assuming this is the package for RAGAS
import random
import string

# Setting API Keys
def set_api_keys():
    OPENAIAPIKEY = st.secrets["OPENAI_API_KEY"]
    LLAMACLOUDAPIKEY = st.secrets["LLAMA_CLOUD_API_KEY"]
    os.environ["OPENAI_API_KEY"] = OPENAIAPIKEY
    os.environ["LLAMA_CLOUD_API_KEY"] = LLAMACLOUDAPIKEY
    return OPENAIAPIKEY

# Parsing PDF to Markdown
def parse_pdf_to_markdown(filepath, output_path):
    parser = LlamaParse(result_type="markdown", num_workers=4, verbose=True, language="en")
    documents = parser.load_data(filepath)
    with open(output_path, 'w', encoding='utf-8') as file:
        for doc in documents:
            if doc.text.strip():
                file.write(doc.text + "\n\n")
    return output_path

# Splitting Markdown into chunks
def split_markdown_into_chunks(md_document_path):
    with open(md_document_path, 'r', encoding='utf-8') as file:
        md_document_content = file.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_chunks = markdown_splitter.split_text(md_document_content)
    return md_header_chunks

# Creating FAISS Retriever
def create_faiss_retriever(md_header_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(md_header_chunks, embeddings)
    return vectorstore.as_retriever()

# Creating RAG Chain
def create_rag_chain(retriever, openai_api_key):
    template = """
    You are an assistance for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer as per the retrieved context.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini")
    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | output_parser
    )
    return rag_chain

# Generating Automatic Prompts using Giskard
def generate_automatic_prompts(md_header_chunks):
    prompt_generator = GiskardPromptGenerator()
    generated_prompts = prompt_generator.generate_prompts(md_header_chunks)
    return generated_prompts

# Evaluating Results with RAGAS
def evaluate_results_with_ragas(question, answer):
    evaluator = RAGASEvaluator()
    score = evaluator.evaluate(question, answer)
    return score

# Asking Questions and Evaluating Responses
def ask_questions(rag_chain, md_header_chunks):
    st.title("PDF CHAT BOT")
    
    # Generate automatic questions
    generated_questions = generate_automatic_prompts(md_header_chunks)
    
    # Loop over each question and get answers
    for question in generated_questions:
        st.write(f"Generated Question: {question}")
        answer = rag_chain.invoke(question)
        st.write(f"Answer: {answer}")
        
        # Evaluate the answer using RAGAS
        score = evaluate_results_with_ragas(question, answer)
        st.write(f"RAGAS Evaluation Score: {score}")

    return generated_questions

# Main function
def main():
    openai_api_key = set_api_keys()

    st.markdown("<h1 style='color:Tomato; text-align: center;'>PDF CHAT BOT</h1>", unsafe_allow_html=True)
    st.markdown("<h6>Ask questions directly to your PDF instead of searching through it for hours</h6>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_pdf:
        local_dir = os.path.expanduser("~/Documents/pdf_to_md_files")
        os.makedirs(local_dir, exist_ok=True)

        pdf_filepath = os.path.join(local_dir, uploaded_pdf.name)
        with open(pdf_filepath, "wb") as f:
            f.write(uploaded_pdf.read())

        markdown_output_path = os.path.join(local_dir, f"{os.path.splitext(uploaded_pdf.name)[0]}.md")

        if os.path.exists(markdown_output_path):
            st.write(f"Markdown file already exists at: {markdown_output_path}. Using the existing file.")
        else:
            st.write("Parsing PDF and generating Markdown file...")
            parse_pdf_to_markdown(pdf_filepath, markdown_output_path)
            st.write(f"Markdown file saved at: {markdown_output_path}")

        md_header_chunks = split_markdown_into_chunks(markdown_output_path)
        retriever = create_faiss_retriever(md_header_chunks, openai_api_key)
        rag_chain = create_rag_chain(retriever, openai_api_key)

        ask_questions(rag_chain, md_header_chunks)

if __name__ == "__main__":
    main()
