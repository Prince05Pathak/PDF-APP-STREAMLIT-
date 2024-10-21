# PDF Chat Bot Function Overview

##### This document provides a breakdown of each function within the PDF Chat Bot application. The bot allows users to upload PDFs, convert them into Markdown, and then ask questions based on the content using a Retrieval-Augmented Generation (RAG) approach.

Models used  for development of this bot :

llama-parse == 0.5.8
datasets == 3.0.1
pandas == 2.2.3
pymongo == 4.10.1 
sentence-transformers == 3.2.0
langchain == 0.3.3
openai == 1.51.2
tiktoken == 0.8.0
langchain-community == 0.3.2
huggingface-hub == 0.25.2
torch == 2.4.1
torchvision == 0.19.1
langchain-openai == 0.2.2
faiss-cpu == 1.9.0
ragas == 0.2.1
streamlit == 1.39.0

# Steps to Use :

  -> Step 01 : Install the requirements.txt for installation of dependecies mentioned above to use this application on your local machine.
            Syntax :  ! pip install -r requirements.txt 

  -> Step 02 : Run the app using Streamlit command to deploy this application on Streamlit servers 
            Syntax : streamlit run script.py 

# Functioning of the code :

  01 -> Reading OpenAI and LLama Cloud Key :
        In script.py the function " set_api_keys() " is used to set the api keys of OPENAI and LLAMA CLOUD for reading the pdf and generating 
        embeddings for the chunks that got created after the parsing of the pdf.

  02 -> Parsing PDF and generating Markdowns :
        The function "def parse_pdf_to_markdown() " is used to read the pdf using the LLAMA PARSE and generating the markdown file's form it from
        the chunks of the pdf are going to be generated

  03 -> Markdown to Chunks generation :
        In function "def split_markdown_into_chunks()" , I have used a model MarkdownHeaderTextSplitter from langchain_text_splitters to divide the data 
        in the pdf on the basis of headings that are mentioned in the parsed data .
        The chunked data formt is mentioned bellow :
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")

  04 -> Generating embeddings :
        The function " def create_faiss_retriever(md_header_chunks, openai_api_key)" creates a FAISS-based retriever that indexes markdown chunks using OpenAI              embeddings   for efficient search and retrieval of relevant information.
  
  05 -> Retrieval-Augmented Generation (RAG) pipeline setup :
        The function "create_rag_chain(retriever, openai_api_key)" Sets up the Retrieval-Augmented Generation (RAG) pipeline to combine the retriever
        with OpenAI’s language model for answering questions.

  06 -> User Query Generation :
        The function "ask_questions(rag_chain)" provides a Streamlit interface for users to input their questions and receive answers from the RAG pipeline.

  07 -> Calling of whole application :
        The function "main()" orchestrates the entire app flow, from uploading a PDF to interacting with the user through questions and answers.

# Conclusion 
This app enables users to upload a PDF, convert it into Markdown, and ask questions based on the document's content using a RAG approach. The process involves extracting content, chunking it, indexing it with FAISS, and generating answers via OpenAI’s GPT model. Each function plays a critical role in the overall pipeline.
