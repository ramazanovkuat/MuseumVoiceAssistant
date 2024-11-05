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

question = "Что такое юрта?"
answer = rag.get_llm_answer(question)

print(answer)