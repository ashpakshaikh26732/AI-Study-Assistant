import streamlit as st
import sys
import yaml
import os
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.rag_core.retriever import create_retriever
from src.features.generator import create_qa_chain
from src.llm.model_loader import load_llm
from src.features.summarizer import create_summarizer_chain
from src.features.flashcard_generator import create_flashcard_chain
from src.features.quiz_engine import grade_user_answer
from src.memory.tracker import initialize_database , log_mistake , get_weak_topics

"""
This script serves as the main entry point for the AI Study Assistant, a
Streamlit-based web application.

The application provides a comprehensive suite of tools to help with studying:
1.  A conversational chat interface for asking questions about study notes.
2.  A "Study Tools" sidebar with advanced features:
    - A topic-based summarizer to get concise overviews.
    - A flashcard generator to create question-and-answer pairs.
    - An interactive quiz mode that uses semantic similarity to grade answers.
3.  A "Learning Profile" section that tracks user mistakes during quizzes and
    provides personalized feedback on topics that need more review, creating an
    adaptive learning experience.

Key functionalities include component caching for performance, session state
management for chat and quiz persistence, and a multi-mode UI that switches
cleanly between different application states.
"""

@st.cache_resource
def load_all_components(config_path="config.yaml"):
@st.cache_resource
def load_all_components(config_path="config.yaml"):
    """
    Loads all major backend components once and caches them for the session.

    This function is decorated with @st.cache_resource to prevent reloading
    the expensive models and chains on every user interaction. It loads the
    LLM, the retriever, all specialized chains (QA, Summarizer, Flashcard),
    the embedding model, and initializes the memory database.

    Args:
        config_path (str): The path to the project's configuration file.

    Returns:
        dict: A dictionary containing all the initialized components required
              for the application to run.
    """
    print("Loading all components... (This should only happen once per session)")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    embedding_model=HuggingFaceEmbeddings(
        model_name=config['rag_core']['embedding']['model_name']
    )
    llm = load_llm(config)
    retriever = create_retriever(config)
    qa_chain = create_qa_chain(retriever, llm, config)
    summarizer_chain = create_summarizer_chain(llm=llm, config=config)
    flashcard_chain = create_flashcard_chain(llm=llm)
    initialize_database(config)
    return {'qa': qa_chain, 'summarizer': summarizer_chain, 'retriever': retriever, 'flashcard_chain': flashcard_chain , 'embedding_model' : embedding_model , 'config': config }


st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI Study Assistant 🤖")


components = load_all_components()
qa_chain = components['qa']
summarizer_chain = components['summarizer']
retriever = components['retriever']
flashcard_chain = components['flashcard_chain']
embedding_model = components['embedding_model']
config = components['config']

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.get('quiz_in_progress', False):

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask me anything about your notes...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke(user_question)
                answer = result['result']
                source_documents = result['source_documents']

                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Show Sources"):
                        for doc in source_documents:
                            st.markdown(f"**Source:** `{doc.metadata.get('source', 'N/A')}`")
                            st.markdown(f"**Content:** {doc.page_content}")
                            st.markdown("---")
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)


    with st.sidebar:
        st.header('Study Tools')

        if retriever:
            vector_store = retriever.vectorstore
            all_docs = vector_store.get()
            all_metadata = all_docs.get('metadatas', [])
            list_of_topics = sorted(list(set(meta.get('course', None) for meta in all_metadata if meta)))
        else:
            list_of_topics = []

        topic = st.selectbox("Select a topic for study tools", list_of_topics)

        if st.button('Generate Summary'):
            if topic:
                with st.spinner(f"Summarizing {topic}..."):
                    try:
                        docs_for_topic = retriever.vectorstore.get(
                            where={'course': topic}
                        )
                        documents_to_summarize = [
                            Document(page_content=text, metadata=meta)
                            for text, meta in zip(docs_for_topic['documents'], docs_for_topic['metadatas'])
                        ]

                        if documents_to_summarize:
                            summary_result = summarizer_chain.invoke(documents_to_summarize)
                            st.subheader(f"Summary for {topic}")
                            st.write(summary_result['output_text'])
                        else:
                            st.warning(f"No documents found for the topic: {topic}")
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")

        if st.button('Generate Flashcards'):
            if topic:
                with st.spinner(f'Generating Flashcards on {topic}...'):
                    try:
                        docs_for_topic = retriever.vectorstore.get(where={'course': topic})
                        list_of_texts = docs_for_topic.get('documents', [])
                        if list_of_texts:
                            combined_context = '\n\n'.join(list_of_texts)
                            flashcard_result = flashcard_chain.invoke({"context": combined_context})
                            st.subheader(f"Flashcards for {topic}")
                            for flashcard in flashcard_result.flashcards:
                                with st.expander(flashcard.question):
                                    st.write(flashcard.answer)
                        else:
                            st.warning("No text content found for this topic.")
                    except Exception as e:
                        st.error(f"An error occurred during flashcard generation: {e}")
        

        if st.button('Start Quiz'):
            if topic :
                with st.spinner(f'Generating Quiz on {topic}...'):
                    try:
                        docs_for_topic = retriever.vectorstore.get(
                            where={'course': topic}
                        )
                        list_of_texts = docs_for_topic.get('documents', [])

                        if list_of_texts:
                            combined_context = '\n\n'.join(list_of_texts)
                            generated_flashcards = flashcard_chain.invoke({"context": combined_context})

                            st.session_state.quiz_questions = generated_flashcards_obj.flashcards
                            st.session_state.current_question_index = 0
                            st.session_state.score = 0
                            st.session_state.quiz_in_progress = True
                            st.session_state.quiz_topic = topic
                            st.rerun()

                        else:
                            st.warning("No text content found for this topic.")
                    except Exception as e:
                        st.error(f"An error occurred during flashcard generation: {e}")
        
        st.divider()
        st.header("Your Learning Profile")
        if st.button("Analyze My Performance"): 
            weak_topics = get_weak_topics(config) 
            if weak_topics : 
                for weak_topic in weak_topics: 
                    st.write(f'-You have have {weak_topic[1]} mistakes in topic - **{weak_topic[0]}**')
            else : 
                st.success("No mistakes logged yet. Keep up the great work!")



else :
    current_q = st.session_state.quiz_questions[st.session_state.current_question_index]
    st.subheader(current_q['question'])
    user_answer = st.text_input("Your Answer")
    if st.button('Submit Answer'):
        result=grade_user_answer(user_answer,current_q['answer'] , embedding_model , config )
        if result :
            st.session_state.score +=1
            st.success('Correct!')
        else :
            st.info(f"Not quite. The correct answer was: {current_q['answer']}")
            log_mistake(st.session_state.quiz_topic ,current_q.question , config)
        st.session_state.current_question_index+=1
        if st.session_state.current_question_index ==len(st.session_state.quiz_questions):
            st.markdown(f'Your Score : {st.session_state.score}')
            st.session_state.quiz_in_progress = False
