import sys
import os
import yaml
import argparse


repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.rag_core.chunker import chunk_single_document
from src.rag_core.embedder import embed_and_store

def main(config):
    """
    Orchestrates the chunking and embedding of all processed documents.

    This function serves as the main logic for the script. It walks through
    the processed data directory, chunks each corrected text file, collects
    all chunks into a master list, and then embeds and stores them in the
    ChromaDB vector store in a single, efficient batch operation.

    Args:
        config (dict): A dictionary containing the configuration loaded
                       from the project's config.yaml file.
    """
    print("Starting the build of the vector store from processed documents...")

    all_chunks = []
    processed_path = config['data']['processed_path']

    for root, dirs, files in os.walk(processed_path):
        for f in files:
            if f.endswith('.txt'):
                full_file_path = os.path.join(root, f)
                print(f"Chunking document: {full_file_path}...")
                

                document_chunks = chunk_single_document(full_file_path, config)
                all_chunks.extend(document_chunks)
    
    if not all_chunks:
        print("No documents found to process. Exiting.")
        return

    print(f"\nAll {len(all_chunks)} document chunks have been created. Starting the embedding process...")


    embed_and_store(all_chunks, config)

    print("\nVector store has been successfully built!")

if __name__ == "__main__":
    """
    Main entry point for the vector store build script.

    This script is designed to be run from the command line after the user
    has completed the manual review and correction of the text files in the
    `data/processed` directory. It handles the final two stages of the
    data pipeline: chunking and embedding.

    The script requires a single command-line argument:
    --config: The path to the project's main configuration YAML file.

    Example usage:
        python build_vector_store.py --config config.yaml
    """
    parser = argparse.ArgumentParser(description='AI Study Assistant - Build Vector Store Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the project config.yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
