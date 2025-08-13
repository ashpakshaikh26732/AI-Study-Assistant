import streamlit as st
import sys
import yaml
import os
from langchain_core.documents import Document


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.rag_core.retriever import create_retriever
from src.features.generator import create_qa_chain
from src.llm.model_loader import load_llm
from src.features.summarizer import create_summarizer_chain

"""
This script serves as the main entry point for the AI Study Assistant, a
Streamlit-based web application.

The application provides two primary features:
1.  A conversational chat interface where users can ask questions about their
    study notes. The backend uses a Retrieval-Augmented Generation (RAG)
    pipeline to provide answers and cite sources from the user's documents.
2.  A "Study Tools" sidebar that includes additional functionalities, starting
    with a topic-based summarizer.

Key functionalities of this script include:
-   **Component Caching:** It uses Streamlit's `@st.cache_resource` to load all
    heavy components (LLM, retriever, and chains) only once per session,
    ensuring the app is responsive after initial startup.
-   **Session State Management:** It leverages `st.session_state` to maintain
    the history of the chat conversation, allowing for a continuous and
    stateful user experience.
-   **Dynamic UI Generation:** The chat history is dynamically rendered on each
    script rerun. The sidebar tools dynamically populate options (like the
    list of topics) by querying the vector store's metadata.
-   **Backend Orchestration:** It handles the user's interaction by invoking the
    appropriate backend chain (QA or Summarizer) and displaying the results,
    including spinners for loading states and expanders for additional details
    like source documents.
"""

@st.cache_resource
def load_all_components(config_path="config.yaml"):
    """
    Loads all major backend components once and caches them for the session.

    This function is decorated with @st.cache_resource to prevent reloading
    the expensive models and chains on every user interaction. It loads the
    LLM, the retriever, the QA chain, and the summarizer chain.

    Args:
        config_path (str): The path to the project's configuration file.

    Returns:
        dict: A dictionary containing the initialized 'qa' chain,
              'summarizer' chain, and the 'retriever' object.
    """
    print("Loading all components... (This should only happen once per session)")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm = load_llm(config)
    retriever = create_retriever(config)
    qa_chain = create_qa_chain(retriever, llm, config)
    summarizer_chain = create_summarizer_chain(llm=llm, config=config)
    return {'qa': qa_chain, 'summarizer': summarizer_chain, 'retriever': retriever}


st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI Study Assistant 🤖")


components = load_all_components()
qa_chain = components['qa']
summarizer_chain = components['summarizer']
retriever = components['retriever']


if "messages" not in st.session_state:
    st.session_state.messages = []

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

    topic = st.selectbox("Select topic to summarize", list_of_topics)

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
