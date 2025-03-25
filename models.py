from langchain_community.document_loaders import UnstructuredFileLoader 
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS, chroma
from langchain.memory import ConversationBufferMemory
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
def create_vector_store(file_paths):
    embeddings = OllamaEmbeddings(model=embedding_model)
    all_chunked_docs = []
    for file_path in file_paths:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)
        all_chunked_docs.extend(chunked_docs)
    db = FAISS.from_documents(all_chunked_docs, embeddings)
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
    #上下文記憶
    memory = ConversationBufferMemory()
    memory_context = memory.load_memory_variables({})["history"]
    full_context = f"{memory_context}\n\nRetrieved Context: {context}"
    model = OllamaLLM(model=llm_model)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": full_context})
    memory.save_context({"input": question}, {"output": response})
    sentences = response.split(". ")[:3]
    return ". ".join(sentences) + "."