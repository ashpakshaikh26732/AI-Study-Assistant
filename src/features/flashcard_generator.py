from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Flashcard(BaseModel):
    """A single flashcard with a question and an answer."""
    question: str = Field(description="The question for the flashcard.")
    answer: str = Field(description="The answer to the flashcard's question.")

class Flashcards(BaseModel):
    """A list of flashcard objects."""
    flashcards: List[Flashcard] = Field(description="A list of flashcards generated from the text.")

def get_flashcard_prompt():
    """
    Creates and returns the prompt template for the flashcard generation chain.
    """
    prompt_string = """
You are an expert educator and study assistant. Your task is to analyze the provided text from a student's notes and generate a series of flashcards to help them study.

Based on the context below, create a list of clear and concise question-and-answer pairs that cover the most important concepts, definitions, and key facts in the text.

{format_instructions}

Context:
---
{context}
---
"""
    prompt_template = PromptTemplate(
        template=prompt_string,
        input_variables=["context"],

        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=Flashcards).get_format_instructions()}
    )
    return prompt_template

def create_flashcard_chain(llm):
    """
    Builds and returns a chain that generates flashcards from a given context.

    The chain takes a block of text (context), formats it with a specialized
    prompt, sends it to the language model, and then uses a JsonOutputParser
    to convert the LLM's string output into a structured Python object.

    Args:
        llm (langchain_core.language_models.base.BaseLanguageModel): The
            initialized language model object (e.g., HuggingFacePipeline).

    Returns:
        langchain.chains.base.Chain: A fully configured chain that, when
        invoked with a context, returns a Pydantic 'Flashcards' object.
    """

    parser = JsonOutputParser(pydantic_object=Flashcards)
    

    prompt = get_flashcard_prompt()

    flashcard_chain = prompt | llm | parser
    
    return flashcard_chain

