# llm_rag_streamlit

1. Set up virtual environment
python3.12 -m venv venv
source venv/bin/activate

2. install requirements
pip install -r requirements.txt

3. Download the ollama model
ollama pull deepseek-r1:1.5b

4. run the program
streamlit run main.py


"""if uploaded_file is not None:
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
    st.info("Please upload a file to start.")"""