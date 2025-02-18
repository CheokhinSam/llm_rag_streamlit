import streamlit as st
import models as models

st.title("Chat with PDFs with Deepseek")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    models.upload_pdf(uploaded_file)
    db = models.create_vector_store(models.pdfs_directory + uploaded_file.name)
    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = models.retrieve_docs(db, question)
        answer = models.question_pdf(question, related_documents)
        st.chat_message("assistant").write(answer)
