import streamlit as st
import models as models

st.title("RAG project with DeepSeek-r1:1.5B")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload the file by using the uploader.
    2. Ask questions related to the document.
    3. The system will retrieve content and provide a concise answer.""")

    st.header("Settings")
    st.markdown("""
    - LLM : DeepSeek-R1:1.5B
    - Embedding Model : DeepSeek-R1:1.5B
    - Retriever Type : Similarity Search
    """)

st.header("Upload a File")
uploaded_file = st.file_uploader(
    "Upload your file here",
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.success("File uploaded, Please wait for processing...")
    models.upload_file(uploaded_file)
    db = models.create_vector_store(models.WORKING_DIR + uploaded_file.name)

    st.markdown("<h2 style='text-align: center;'>Teacher Assistant Chatbot</h2>", unsafe_allow_html=True)
    question = st.chat_input("Ask a Question here!")

    if question:
        st.chat_message("user").write(question)
        related_documents = models.retrieve_docs(db, question)
        answer = models.question_file(question, related_documents)
        st.chat_message("assistant").write(answer)
else:
    st.info("Please upload a file to start.")