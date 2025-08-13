import os,sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)


def chunk_single_document(file_path, config):
    """
    Reads a single text document, splits it into smaller chunks, and
    attaches metadata to each chunk.

    This function takes a file path to a processed .txt file, reads its
    content, and uses a RecursiveCharacterTextSplitter from LangChain to
    divide the text into manageable pieces based on the chunk size and
    overlap specified in the configuration. It also extracts metadata
    (like course, topic, etc.) from the file path and includes the
    original source path.

    Args:
        file_path (str): The full path to the .txt file to be chunked.
        config (dict): The project's configuration dictionary, which must
                     contain chunk_size and chunk_overlap parameters under
                     the 'rag_core.chunking' key.

    Returns:
        list[langchain_core.documents.base.Document]: A list of LangChain
        Document objects. Each Document object represents a single chunk
        and contains the chunk's text (`page_content`) and its associated
        metadata dictionary.
    """
    document_text = ''
    with open(file_path, 'r', encoding='utf-8') as doc:
        content = doc.read()
        document_text += content
    
    meta_data = {}
    path_data = file_path.split(os.sep)
    path_data = path_data[2:]

    if len(path_data) > 2:
        meta_data = {
            'source': file_path,
            'specialization': path_data[0],
            'course': path_data[1],
            'notes_type': path_data[2]
        }
    elif len(path_data) == 2:
        meta_data = {
            'source': file_path,
            'course': path_data[0],
            'notes_type': path_data[2]
        }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['rag_core']['chunking']['chunk_size'],
        chunk_overlap=config['rag_core']['chunking']['chunk_overlap'],
        separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.create_documents(
        texts=[document_text],
        metadatas=[meta_data]
    )
    return documents
