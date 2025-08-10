import sys
import argparse
import yaml

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

# !pip install -q monai
from monai.apps import download_and_extract

from src.rag_core.retriever import create_retriever 
from src.features.generator import create_qa_chain

parser = argparse.ArgumentParser(description='parsing qa script for ai study assistant')
parser.add_argument('--config' , type = str , required = True , help = 'path to yaml file')
args = parser.parse_args()
with open(args.config , 'r') as f : 
    config = yaml.safe_load(f)

retriever =create_retriever(config)
qa_chain = create_qa_chain(retriever , config)

query = "What is a Bi-directional RNN?"
result = qa_chain.invoke(query)
print("Answer:", result['result'])
for doc in result['source_documents']:
    print(f"page_content : {doc['page_content']} ")
    print(f"metadata' : {doc['metadata']}")
