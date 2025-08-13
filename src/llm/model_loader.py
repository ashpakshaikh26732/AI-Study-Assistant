import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

def load_llm(config):
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
    return llm