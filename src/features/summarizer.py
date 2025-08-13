import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def get_summarizer_prompts():
    """
    Creates and returns the prompt templates for the map-reduce summarization chain.

    This function defines two distinct prompts:
    1. A 'map' prompt to extract key points from individual document chunks.
    2. A 'combine' prompt to synthesize a final summary from the collection
       of key points.

    Returns:
        tuple[PromptTemplate, PromptTemplate]: A tuple containing the
        map_prompt_template and the combine_prompt_template.
    """

    map_prompt_string = """
You are an expert academic assistant skilled at distilling complex information. Your task is to analyze the following text from a student's notes and extract the most critical information.

Focus on identifying and clearly stating the main concepts, key definitions, important formulas, and core principles. Ignore any filler text, examples, or conversational parts. Present the output as a concise list of key points.

Text:
"{text}"

Concise Key Points:
"""
    map_prompt_template = PromptTemplate(
        template=map_prompt_string,
        input_variables=["text"]
    )

    combine_prompt_string = """
You are a master of synthesis, tasked with creating a final, high-quality summary from a collection of key points extracted from a student's notes.

Your goal is to weave these individual points into a single, coherent, and well-organized summary. The final output should be easy to read, logically structured, and cover all the essential information from the provided points. Start with a brief overview, then elaborate on the key topics.

Collection of Key Points:
"{text}"

Comprehensive Final Summary:
"""
    combine_prompt_template = PromptTemplate(
        template=combine_prompt_string,
        input_variables=["text"]
    )
    
    return map_prompt_template, combine_prompt_template

def create_summarizer_chain(llm, config):
    """
    Builds and returns a summarization chain using the map-reduce strategy.

    This function reuses the provided language model (LLM) and constructs a
    specialized chain for summarization. It uses custom prompts for both the
    'map' and 'combine' steps to ensure high-quality, structured summaries.

    Args:
        llm (langchain_core.language_models.base.BaseLanguageModel): The
            initialized language model object (e.g., HuggingFacePipeline)
            that will be used for summarization.
        config (dict): The project's configuration dictionary. This is included
                     for potential future use, such as adding specific
                     summarizer settings.

    Returns:
        langchain.chains.base.Chain: A fully configured summarization chain
        object, ready to be invoked with a list of documents.
    """
    map_prompt_template, combine_prompt_template = get_summarizer_prompts()

    summarization_chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True
    )

    return summarization_chain
