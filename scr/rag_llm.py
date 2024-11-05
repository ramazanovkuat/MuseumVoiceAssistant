import os
import pickle
from typing import Any, Literal
from langchain.docstore.document import Document
from langchain_community.llms import YandexGPT

from dotenv import load_dotenv
load_dotenv()

class RAG:
    """
    Класс для реализации Retrieval-Augmented Generation (RAG),
    который загружает ретривер и использует его для получения ответов на вопросы.
    """
    
    def __init__(self, retriever_path_to_upload: str,
                 model_name: Literal['yandexgpt', 'yandexgpt-lite', 'yandexgpt-32k']='yandexgpt-lite',
                 model_version: Literal['deprecated', 'latest', 'rc']='latest'):
        """
        Инициализирует экземпляр RAG, загружая ретривер из файла.
        Подробнее о версиях моделей: 
        https://yandex.cloud/ru/docs/foundation-models/concepts/yandexgpt/models

        Примечаение: модель 'yandexgpt-32k' только в версии 'rc' на 05.11.2024.

        Args:
            retriever_path_to_upload (str): путь к ретриверу (с расширением).
            model_name (Literal['yandexgpt', 'yandexgpt-lite', 'yandexgpt-32k']): название модели
            model_version (Literal['deprecated', 'latest', 'rc']): версия модели
        """
        self.retriever_path_to_upload = retriever_path_to_upload
        self.retriever = self.upload_retriever()
        self.yandex_gpt = YandexGPT(api_key=os.getenv('YGPT_API_KEY'), 
                                    folder_id=os.getenv('YGPT_FOLDER_IP'), 
                                    model_name=model_name,
                                    model_version=model_version)
    
    def upload_retriever(self) -> Any:
        """
        Загружает ретривер из файла .pkl и возвращает его.

        Returns:
            Any: объект ретривера.
        """
        with open(self.retriever_path_to_upload, 'rb') as f:
            retriever_uploaded = pickle.load(f)
        return retriever_uploaded
    
    def get_documents(self, question: str) -> list[Document]:
        """
        Получает документы, связанные с вопросом, из ретривера.

        Args:
            question (str): вопрос, для которого необходимо извлечь документы.

        Returns:
            list[Document]: документы, извлеченные ретривером.
        """
        documents = self.retriever.invoke(question)
        return documents
    
    def get_llm_answer(self, question: str) -> str:
        """
        Генерирует ответ на вопрос, используя поисковую выдачу и LLM.

        Args:
            question (str): вопрос, для которого требуется сгенерировать ответ.

        Returns:
            str: ответ, сгенерированный с использованием поисковой выдачи и LLM.
        """
        documents = self.get_documents(question)
        context = self.process_retrieve_output(documents)
        
        prompt = (
            "Ты - система генерации ответа на основе поисковой выдачи. Тебе дан вопрос и поисковая выдача. "
            "Создай КРАТКИЙ и ИНФОРМАТИВНЫЙ ответ (не более 50 слов) на заданный вопрос, "
            "основываясь на приведенной поисковой выдаче или собственных знаниях. Не повторяй текст. "
            "Ответ сформируй как простой текст, не Markdown. Используй тон музейного экскурсовода. "
            f"в поисковой выдаче, содержание поисковой выдачи - по ключу 'содержание'. Вопрос: {question}. "
            f"Поисковая выдача: {context}"
        )
        
        answer = self.yandex_gpt.invoke(prompt)
        answer_processer = self.process_answer(answer)
        return answer_processer
    
    def process_data(self, doc: Document) -> dict:
        """Редактирование документа, полученного из ретривера

        Args:
            doc (Document): документ из ретривера

        Returns:
            dict: словарь с содержанием 'содержание' и названием источника 'источник'
        """
        page_content = doc.page_content
        if 'source' in doc.metadata.keys():
            source = doc.metadata['source'].split("/")[-1].split(".")[0]
        else:
            source = ''
        return {'источник': source, 'содержание': page_content}

    def process_retrieve_output(self, docs: list[Document]) -> list[dict]:
        """Функция обработки документов из ретривера.

        Args:
            docs (list[Document]): документы из ретривера

        Returns:
            list[dict]: обработанные документы
        """
        processed_documents = []
        for doc in docs:
            processed_doc = self.process_data(doc)
            processed_documents.append(processed_doc)
        return processed_documents
    
    def process_answer(self, answer: str) -> str:
        """Обработка ответа от LLM.

        Args:
            answer (str): ответ от LLM

        Returns:
            str: обработанный ответ от LLM
        """
        # Убираем перенос строки ("\n") и пробелы в начале и конце строки.
        answer_processed = answer.replace("\n", " ").strip()
        return answer_processed