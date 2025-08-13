import streamlit as st
import sys
import yaml


repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.rag_core.retriever import create_retriever
from src.features.generator import create_qa_chain
from src.llm.model_loader import load_llm

@st.cache_resource
def load_all_components(config_path="config.yaml"):
    """
    Loads the QA chain once and caches it for the entire session.

    This function is decorated with @st.cache_resource, which means it
    runs only once when the app starts, loading the models and building
    the QA chain. The result is then stored in memory, making subsequent
    calls instantaneous.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        langchain.chains.retrieval_qa.base.RetrievalQA: The fully
        configured QA chain object.
    """
    print("Loading QA chain... (This should only happen once per session)")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm = load_llm(config)
    retriever = create_retriever(config)
    qa_chain = create_qa_chain(retriever, config)
    return qa_chain


st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI Study Assistant 🤖")


qa_chain = load_all_components()

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

