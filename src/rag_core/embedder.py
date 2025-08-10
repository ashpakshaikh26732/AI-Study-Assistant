import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def embed_and_store(documents, config):
    """
    Embeds a list of text documents and stores them in a persistent
    ChromaDB vector store.

    This function takes a list of LangChain Document objects, initializes a
    Hugging Face embedding model specified in the configuration, and then
    uses LangChain's Chroma class to perform the embedding and storage in
    a single operation. The resulting vector database is saved to the
    directory specified in the configuration, making it persistent.

    Args:
        documents (list[langchain_core.documents.base.Document]): A list of
            LangChain Document objects to be embedded and stored. Each
            document should contain page_content and metadata.
        config (dict): The project's configuration dictionary. It must
            contain the embedding model name, the database persist
            directory, and the collection name under the 'rag_core' key.

    Side Effects:
        - Downloads the specified Hugging Face embedding model if it's not
          already cached.
        - Creates or updates a ChromaDB database at the location specified
          by `persist_directory` in the config.
        - Prints status messages from the underlying libraries to the console.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config['rag_core']['embedding']['model_name']
    )
    
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config['rag_core']['database']['persist_directory'],
        collection_name=config['rag_core']['database']['collection_name']
    )

