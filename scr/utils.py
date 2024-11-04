from langchain.docstore.document import Document

def process_data(doc: Document) -> dict:
    """Редактирование документа, полученного из ретривера

    Args:
        doc (Document): документ из ретривера

    Returns:
        dict: словарь с содержанием 'содержание' и названием источника 'источник'
    """
    page_content = doc.page_content
    source = doc.metadata['source'].split("/")[-1].split(".")[0]
    return {'источник': source, 'содержание': page_content}

def process_retrieve_output(docs: list[Document]) -> list[dict]:
    """Функция обработки документов из ретривера.

    Args:
        docs (list[Document]): документы из ретривера

    Returns:
        list[dict]: обработанные документы
    """
    processed_documents = []
    for doc in docs:
        processed_doc = process_data(doc)
        processed_documents.append(processed_doc)
    return processed_documents