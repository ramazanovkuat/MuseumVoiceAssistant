import os 
from rag_llm import RAG

import streamlit as st
st.title("ИИ-экскурсовод")

# Определение пути к векторной базе данных
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
data_folder = os.path.join(parent_dir, "data")
# TODO: вынести в config-файл название векторной базы данных
vector_db_path = os.path.join(data_folder, "vector_dbs", "bm25_faiss_rerank_first.pkl")

# Инициализация экземпляра класса RAG
rag = RAG(retriever_path_to_upload=vector_db_path)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Введите вопрос"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        answer = rag.get_llm_answer(question)
        st.markdown(answer)