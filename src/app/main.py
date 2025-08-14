import streamlit as st
import sys
import yaml
import os
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from streamlit_mic_recorder import mic_recorder


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.rag_core.retriever import create_retriever
from src.features.generator import create_qa_chain
from src.llm.model_loader import load_llm
from src.features.summarizer import create_summarizer_chain
from src.features.flashcard_generator import create_flashcard_chain
from src.features.quiz_engine import grade_user_answer
from src.memory.tracker import initialize_database, log_mistake, get_weak_topics
from src.voice.speech_to_text import load_whisper_model, transcribe_audio
from src.voice.text_to_speech import convert_text_to_speech

"""
This script serves as the main entry point for the AI Study Assistant, a
Streamlit-based web application.

The application provides a comprehensive suite of tools to help with studying:
1.  A multi-modal conversational chat interface where users can ask questions
    about their study notes using either text or voice.
2.  A "Study Tools" sidebar with advanced features:
    - A topic-based summarizer to get concise overviews.
    - A flashcard generator to create question-and-answer pairs.
    - An interactive quiz mode that uses semantic similarity to grade answers.
3.  A "Learning Profile" section that tracks user mistakes during quizzes and
    provides personalized feedback on topics that need more review.
4.  Optional voice responses for a fully conversational experience.

Key functionalities include component caching for performance, session state
management for chat and quiz persistence, and a multi-mode UI that switches
cleanly between different application states.
"""

@st.cache_resource
def load_all_components(config_path="config.yaml"):
    """
    Loads all major backend components once and caches them for the session.
    """
    print("Loading all components... (This should only happen once per session)")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    embedding_model = HuggingFaceEmbeddings(
        model_name=config['rag_core']['embedding']['model_name']
    )
    llm = load_llm(config)
    retriever = create_retriever(config)
    qa_chain = create_qa_chain(retriever, llm, config)
    summarizer_chain = create_summarizer_chain(llm=llm, config=config)
    flashcard_chain = create_flashcard_chain(llm=llm)
    initialize_database(config)
    whisper_model = load_whisper_model(config)

    return {
        'qa': qa_chain, 
        'summarizer': summarizer_chain, 
        'retriever': retriever, 
        'flashcard_chain': flashcard_chain, 
        'embedding_model': embedding_model, 
        'config': config, 
        'whisper_model': whisper_model
    }

def handle_user_query(question_text, voice_enabled):
    """
    Handles the processing of a user's query, whether from text or voice.
    """
    if question_text:
        st.session_state.messages.append({"role": "user", "content": question_text})
        with st.chat_message("user"):
            st.markdown(question_text)

        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke(question_text)
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
                
                if voice_enabled:
                    audio_bytes = convert_text_to_speech(answer)
                    if audio_bytes:
                        st.audio(audio_bytes, autoplay=True)

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)


st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI Study Assistant 🤖")


components = load_all_components()
qa_chain = components['qa']
summarizer_chain = components['summarizer']
retriever = components['retriever']
flashcard_chain = components['flashcard_chain']
embedding_model = components['embedding_model']
config = components['config']
whisper_model = components['whisper_model']


with st.sidebar:
    st.header("Settings & Tools")
    voice_enabled = st.toggle("Enable Voice Responses")
    st.divider()
    
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
                    docs_for_topic = retriever.vectorstore.get(where={'course': topic})
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
        if topic:
            with st.spinner(f'Generating Quiz on {topic}...'):
                try:
                    docs_for_topic = retriever.vectorstore.get(where={'course': topic})
                    list_of_texts = docs_for_topic.get('documents', [])
                    if list_of_texts:
                        combined_context = '\n\n'.join(list_of_texts)
                        generated_flashcards = flashcard_chain.invoke({"context": combined_context})
                        
                        st.session_state.quiz_questions = generated_flashcards.flashcards
                        st.session_state.current_question_index = 0
                        st.session_state.score = 0
                        st.session_state.quiz_in_progress = True
                        st.session_state.quiz_topic = topic
                        st.rerun()
                    else:
                        st.warning("No text content found for this topic.")
                except Exception as e:
                    st.error(f"An error occurred during quiz generation: {e}")
    
    st.divider()
    st.header("Your Learning Profile")
    if st.button("Analyze My Performance"):
        weak_topics = get_weak_topics(config)
        if weak_topics:
            st.write("Based on your quiz results, you might want to review these topics:")
            for topic_name, mistake_count in weak_topics:
                st.write(f"- You have {mistake_count} mistakes in the topic: **{topic_name}**")
        else:
            st.success("No mistakes logged yet. Keep up the great work!")


if not st.session_state.get('quiz_in_progress', False):

    st.header("Chat with your Notes")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    col1, col2 = st.columns([10, 1])
    with col1:
        user_question = st.chat_input("Ask me anything about your notes...")
        if user_question:
            handle_user_query(user_question, voice_enabled)
    
    with col2:
        st.write("") 
        st.write("")
        audio_bytes_dict = mic_recorder(key='mic', start_prompt="🎤", stop_prompt="⏹️", just_once=True)
        if audio_bytes_dict:
            transcribed_question = transcribe_audio(audio_bytes_dict['bytes'], whisper_model)
            if transcribed_question:
                handle_user_query(transcribed_question, voice_enabled)

else:

    st.header(f"Quiz on: {st.session_state.get('quiz_topic', 'your notes')}")
    
    if 'quiz_questions' not in st.session_state or not st.session_state.quiz_questions:
        st.warning("No quiz questions found. Please start a new quiz from the sidebar.")
        if st.button("Back to Chat"):
            st.session_state.quiz_in_progress = False
            st.rerun()
    elif st.session_state.current_question_index >= len(st.session_state.quiz_questions):
        st.success(f"Quiz Complete! Your final score: {st.session_state.score}/{len(st.session_state.quiz_questions)}")
        if st.button("Back to Chat"):
            st.session_state.quiz_in_progress = False
            st.rerun()
    else:
        current_q = st.session_state.quiz_questions[st.session_state.current_question_index]
        
        st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.quiz_questions)}")
        st.markdown(f"**{current_q.question}**")
        
        user_answer = st.text_input("Your Answer", key=f"user_answer_{st.session_state.current_question_index}")

        if st.button("Submit Answer"):
            is_correct = grade_user_answer(user_answer, current_q.answer, embedding_model, config)
            
            if is_correct:
                st.session_state.score += 1
                st.success("Correct!")
            else:
                st.error(f"Not quite. The correct answer is: {current_q.answer}")
                log_mistake(st.session_state.quiz_topic, current_q.question, config)
            
            st.session_state.current_question_index += 1
            st.rerun()
