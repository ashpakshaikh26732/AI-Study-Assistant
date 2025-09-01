import sys
import os
import yaml
import argparse

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.preprocessing.document_parser import process_all_documents

def main(config):
    """
    Orchestrates the automated preprocessing of all raw documents.

    This function serves as the main logic for the script. It prints status
    messages to the console and calls the `process_all_documents` function,
    which performs the heavy lifting of extracting and cleaning text from
    all PDFs found in the raw data directory.

    Args:
        config (dict): A dictionary containing the configuration loaded
                       from the project's config.yaml file.
    """
    print("Starting automated preprocessing of all raw documents...")
    process_all_documents(config)
    print("\nAutomated preprocessing complete. Please manually review and correct the files in the data/processed directory.")

if __name__ == "__main__":
    """
    Main entry point for the preprocessing script.

    This script is designed to be run from the command line. It handles the
    first major step of the data pipeline: converting all raw PDF documents
    into cleaned text files. This is the "first draft" generation, which
    prepares the text for the essential manual review and correction step.

    The script requires a single command-line argument:
    --config: The path to the project's main configuration YAML file.

    Example usage:
        python run_preprocessing.py --config config.yaml
    """
    parser = argparse.ArgumentParser(description='AI Study Assistant - Raw Data Preprocessing Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the project config.yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
