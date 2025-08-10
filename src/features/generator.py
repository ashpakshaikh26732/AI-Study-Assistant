import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

def create_qa_chain(retriever, config):
    """
    Builds and returns a complete question-answering (QA) chain using the RAG
    (Retrieval-Augmented Generation) pattern.

    This function performs several key steps:
    1.  Sets up a 4-bit quantization configuration using BitsAndBytes to
        load a large language model efficiently with reduced memory usage.
    2.  Loads the specified tokenizer and the quantized language model from
        Hugging Face.
    3.  Creates a text-generation pipeline using the loaded model and tokenizer,
        configured with parameters for generation like temperature and top_p.
    4.  Wraps the transformers pipeline in a LangChain-compatible
        `HuggingFacePipeline` object.
    5.  Constructs a `RetrievalQA` chain that connects the provided retriever
        and the language model, ready to answer questions based on retrieved
        context.

    Args:
        retriever (langchain_core.vectorstores.VectorStoreRetriever): An
            initialized and configured retriever object responsible for
            fetching relevant documents from the vector store.
        config (dict): The project's configuration dictionary, which must
                     contain the Hugging Face model name for the generator
                     LLM under the 'rag_core.generator.llm_name' key.

    Returns:
        langchain.chains.retrieval_qa.base.RetrievalQA: A fully configured
        QA chain object. This chain can be invoked with a query to get a
        generated answer and the source documents used.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config['rag_core']['generator']['llm_name'])
    model = AutoModelForCausalLM.from_pretrained(
        config['rag_core']['generator']['llm_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        return_source_documents=True
    )
    
    return qa_chain
