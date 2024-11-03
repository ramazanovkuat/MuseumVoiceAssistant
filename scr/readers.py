import warnings
from pathlib import Path, PosixPath
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader


def read_txt_file(path: str | PosixPath) -> Document:
    """Функция чтения txt-файлов. Возвращает докумнент в Document (langchain) формате.

    Args:
        path (str | PosixPath): путь к файлу txt-формата

    Returns:
        Document: прочитанный файл в формате Document
    """
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, PosixPath):
        raise ValueError("path должен быть 'str' или объектом 'Path'")
    
    file_extension = path.suffix.lower()
    if file_extension == '.txt':
        loader = TextLoader(path)
        raw_document = loader.load()
        return raw_document
    else:
        raise ValueError(f"Расширение {file_extension} не поддерживается")
    
def get_raw_documents(folder: PosixPath) -> list[Document]:
    
    """Функция итерации по папке с документами
       Возвращает список с документами в Document (langchain) формате.
       
    Args:
        folder (PosixPath): папка с документами в txt формате

    Returns:
        list: список с документами в Document (langchain) формате
    """

    raw_documents = []

    for file in folder.iterdir():
        file_extension = file.suffix.lower()
        if file_extension == '.txt':
            raw_document = read_txt_file(file)
            raw_documents.extend(raw_document)
        else:
            warnings.warn(f"Чтение {file} не поддерживается.", UserWarning)    
    return raw_documents