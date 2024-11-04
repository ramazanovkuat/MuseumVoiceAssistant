from scr.readers import *
from pathlib import PosixPath
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

def create_bm25_faiss_rerank(documents_folder: str | PosixPath,
                             chunking_method: RecursiveCharacterTextSplitter,
                             weights: list | tuple = [0.5, 0.5],
                             retrieve_n_docs: int = 5,
                             rerank_n_docs: int = 3) -> ContextualCompressionRetriever:
    """Создание ретривера типа EnsembleRetriever с BM25, FAISS и CrossEncoder-реранкером.
        - BM25:
        -- from langchain.retrievers import BM25Retriever
        -- bm25_retriever = BM25Retriever.from_documents(chunked_documents_with_page_content)
            bm25_retriever.k = retrieve_n_docs

        - FAISS:
        -- from langchain_community.vectorstores import FAISS
        -- FAISS.from_texts(chunked_documents, embedding_model, distance_strategy=DistanceStrategy.COSINE)

        - vector_database:
        -- from langchain.retrievers import EnsembleRetriever
        -- vector_database = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=wheights)
            
        - reranker:
        -- from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        -- reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        -- from langchain.retrievers.document_compressors import CrossEncoderReranker
        -- compressor = CrossEncoderReranker(model=reranker, top_n=rerank_n_docs)

    Args:
        documents_folder (str | PosixPath): путь к документам txt-формата.
        chunking_method (RecursiveCharacterTextSplitter): метод чанкинга.
        weights (list | tuple, optional): Веса EnsembleRetriever [bm25, faiss]. Defaults to [0.5, 0.5].
        retrieve_n_docs (int, optional): Количество документов после ретрива. Defaults to 5.
        rerank_n_docs (int, optional): Количесто документов после реранкинга. Defaults to 3.
        
    Returns:
        (ContextualCompressionRetriever) : ретривер-реранкер 
    """
    if isinstance(documents_folder, str):
        documents_folder = Path(documents_folder)
    raw_documents = get_raw_documents(documents_folder)
    chunked_documents_with_page_content = chunking_method.split_documents(raw_documents)
    chunked_documents = [doc.page_content for doc in chunked_documents_with_page_content]
    
    bm25_retriever = BM25Retriever.from_documents(
        chunked_documents_with_page_content)
    bm25_retriever.k = retrieve_n_docs   
    
    # TODO: сделать более универсально
    model_name = "deepvk/USER-bge-m3"
    device='cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device})
    # TODO: сдлеать возможность загрузки готовой db из локально сохраненных
    db = FAISS.from_texts(chunked_documents, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    retriever = db.as_retriever(search_kwargs={"k": retrieve_n_docs})
    vector_database = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=weights)
    rerank_model_name = "BAAI/bge-reranker-v2-m3"
    rerank_model = HuggingFaceCrossEncoder(model_name=rerank_model_name)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=rerank_n_docs)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_database)
    return compression_retriever
