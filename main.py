import streamlit as st
import models as models
from pathlib import Path

st.set_page_config(page_title="RAG Q&A System", layout="wide")

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

st.header("Upload Files")
uploaded_files = st.file_uploader(
    "Upload your files here",
    accept_multiple_files=True,  # 允許上傳多個檔案
    help="You can upload multiple files at once"
)



# 處理上傳檔案
if uploaded_files:
    try:
        # 顯示進度提示並處理多個檔案
        with st.spinner("Files uploaded, please wait for processing..."):
            file_paths = []
            for uploaded_file in uploaded_files:
                models.upload_file(uploaded_file)
                file_path = Path(models.WORKING_DIR) / uploaded_file.name
                file_paths.append(str(file_path))
            
            # 假設 create_vector_store 可以處理多個檔案路徑，若不行需調整 models 模組
            db = models.create_vector_store(file_paths)
        st.success(f"{len(uploaded_files)} file(s) processed successfully!")

        # 聊天介面
        st.markdown("<h2 style='text-align: center;'>Teacher Assistant Chatbot</h2>", unsafe_allow_html=True)
        
        # 使用 session_state 保存對話歷史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # 顯示歷史訊息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # 接收新問題
        if question := st.chat_input("Ask a question here!"):
            # 顯示使用者問題
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})

            # 生成並顯示回答
            with st.spinner("Generating answer..."):
                related_docs = models.retrieve_docs(db, question)
                answer = models.question_file(question, related_docs)
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
        st.info("Please check the file formats or try again later.")
else:
    st.info("Please upload at least one file to start.")