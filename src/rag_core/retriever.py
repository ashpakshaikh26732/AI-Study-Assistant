import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_retriever(config):
    """
    Creates a retriever object from a pre-existing, persistent ChromaDB
    vector store.

    This function initializes the same Hugging Face embedding model that was
    used for storing the data. It then connects to the ChromaDB database
    persisted on disk and creates a retriever object from it. The retriever
    is configured with search parameters, such as 'k' for the number of
    documents to return, based on the provided configuration.

    Args:
        config (dict): The project's configuration dictionary. It must
                     contain the embedding model name, the database persist
                     directory, the collection name, and retriever settings
                     (like 'k') under the 'rag_core' key.

    Returns:
        langchain_core.vectorstores.VectorStoreRetriever: A configured
        retriever object ready to be used for fetching relevant documents
        from the vector store in response to a query.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config['rag_core']['embedding']['model_name']
    )
    
    vector_store = Chroma(
        collection_name=config['rag_core']['database']['collection_name'],
        embedding_function=embeddings,
        persist_directory=config['rag_core']['database']['persist_directory']
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": config['rag_core']['retriever']['k']}
    )
    
    return retriever
