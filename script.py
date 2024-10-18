import os
import streamlit as st
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS  
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def set_api_keys():
    OPENAIAPIKEY = "YOUR_OPENAI_API_KEY"
    LLAMACLOUDAPIKEY = "YOUR_LLAMA_CLOUD_API_KEY"
    os.environ["LLAMA_CLOUD_API_KEY"] = LLAMACLOUDAPIKEY
    os.environ["OPENAI_API_KEY"] = OPENAIAPIKEY
    return OPENAIAPIKEY

def parse_pdf_to_markdown(filepath, output_path):
    parser = LlamaParse(result_type="markdown", num_workers=4, verbose=True, language="en")
    documents = parser.load_data(filepath)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for doc in documents:
            if doc.text.strip():
                file.write(doc.text + "\n\n")
    return output_path

def split_markdown_into_chunks(md_document_path):
    with open(md_document_path, 'r', encoding='utf-8') as file:
        md_document_content = file.read()

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_chunks = markdown_splitter.split_text(md_document_content)
    
    return md_header_chunks

def create_faiss_retriever(md_header_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(md_header_chunks, embeddings)
    return vectorstore.as_retriever()

def create_rag_chain(retriever, openai_api_key):
    template = """
    You are an assistant for question-answering tasks.
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

def ask_questions(rag_chain):
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question:
            answer = rag_chain.invoke(question)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")

def main():
    openai_api_key = set_api_keys()

    st.title("PDF CHAT BOT")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_pdf:
        local_dir = os.path.expanduser("~/Documents/pdf_to_md_files")
        os.makedirs(local_dir, exist_ok=True)

        pdf_filepath = os.path.join(local_dir, uploaded_pdf.name)
        with open(pdf_filepath, "wb") as f:
            f.write(uploaded_pdf.read())

        markdown_output_path = os.path.join(local_dir, f"{os.path.splitext(uploaded_pdf.name)[0]}.md")

        # Check if the Markdown file already exists
        if os.path.exists(markdown_output_path):
            st.write(f"Using existing Markdown file at: {markdown_output_path}.")
        else:
            st.write("Parsing PDF and generating Markdown file...")
            parse_pdf_to_markdown(pdf_filepath, markdown_output_path)
            st.write(f"Markdown file saved at: {markdown_output_path}")

        # Split the markdown document into chunks
        md_header_chunks = split_markdown_into_chunks(markdown_output_path)

        # Create FAISS retriever from markdown chunks
        retriever = create_faiss_retriever(md_header_chunks, openai_api_key)

        # Create RAG chain for question answering
        rag_chain = create_rag_chain(retriever, openai_api_key)

        # Add a download button for the generated Markdown file
        with open(markdown_output_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        st.download_button(
            label="Download Markdown File",
            data=md_content,
            file_name=os.path.basename(markdown_output_path),
            mime="text/markdown"
        )

        ask_questions(rag_chain)

if __name__ == "__main__":
    main()
