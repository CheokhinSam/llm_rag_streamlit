from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredFileLoader 
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS, chroma
import os

WORKING_DIR = 'Docs/'

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Set up LLM model
model = OllamaLLM(model="deepseek-r1:1.5b")
# Set up Embedding model
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
# Upload_file
def upload_file(file):
    with open(WORKING_DIR + file.name, "wb") as f:
        f.write(file.getbuffer())
# Set up Vector Database
def create_vector_store(file_path):
    loader = UnstructuredFileLoader(file_path) 
    documents = loader.load()
    """    if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")"""
    
    
    text_splitter =RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )

    chunked_docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db

# Retrieve Docs
def retrieve_docs(db, query, k=4):
    print(db.similarity_search(query))
    return db.similarity_search(query, k)

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def question_file(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})
