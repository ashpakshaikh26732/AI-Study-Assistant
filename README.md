# AI Study Assistant: Your Personalized, Adaptive Learning Partner

# \<!-- Badges Placeholder -->\<p align="center">\<img src="https\://www\.google.com/search?q=https\://img.shields.io/badge/Python-3.10%252B-blue" alt="Python Version">\<img src="https\://www\.google.com/search?q=https\://img.shields.io/badge/LangChain-Powered-green" alt="LangChain">\<img src="https\://www\.google.com/search?q=https\://img.shields.io/badge/License-MIT-yellow" alt="License">\</p>The ****AI Study Assistant**** is a full-stack, multi-modal application designed to transform your personal study notes into an interactive and adaptive learning experience. Built from the ground up with open-source models and a modular architecture, this project serves as both a powerful study tool and a comprehensive blueprint for building modern AI systems.It ingests your typed and handwritten notes, understands the content, and helps you learn through intelligent Q\&A, summarization, and personalized quizzes that adapt to your learning patterns.

## 🚀 Key Features

# This application is packed with features designed to create a comprehensive and effective study environment.* ****🧠 Conversational Q\&A:**** Ask questions about your notes in natural language and get accurate, context-aware answers. The assistant cites the exact sources from your documents for every answer.

* ****🗣️**** Multi-Modal ****Interaction:**** Interact with the assistant using either typed text or your voice. Enable voice responses to have the answers spoken back to you for a fully conversational experience.

* ****📝 Automated Summarization:**** Select any topic from your notes and receive a concise, AI-generated summary of the key concepts, perfect for quick reviews.

* ****🃏 AI-Generated Flashcards:**** Instantly create a set of question-and-answer flashcards for any topic to practice active recall.

* ****✍️ Interactive Quizzing:**** Test your knowledge with quizzes generated from your notes. The assistant uses semantic similarity to intelligently grade your answers, understanding the meaning beyond exact wording.

* ****🎯 Adaptive Learning & Memory:**** The assistant remembers your mistakes! It logs incorrect quiz answers to a local database, allowing it to provide personalized feedback on your weak topics.

## 🎬 Demo

# __(This__ is where you can embed a short video or GIFs showcasing the application __in action.)__Main Q\&A Interface:\[DEMO GIF OF CHATBOT HERE]Study Tools (Summarizer, Flashcards, Quiz):\[DEMO GIF OF SIDEBAR TOOLS HERE]

## 🛠️ Tech Stack & Architecture

# This project uses a modern, modular tech stack composed of powerful open-source libraries and models.|                          |                                                  |                                                                                                  |
| ------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| ****Component****        | ****Technology / Model****                       | ****Purpose****                                                                                  |
| ****UI Framework****     | Streamlit                                        | To build the interactive, real-time web application interface.                                   |
| ****AI Orchestration**** | LangChain                                        | The core framework used to build and connect all components of the RAG and feature chains.       |
| ****Data Ingestion****   | PyMuPDF, Pytesseract                             | For extracting text from both typed and scanned/handwritten PDF documents.                       |
| ****Vector Database****  | ChromaDB                                         | A local, persistent vector store for efficient semantic search and metadata filtering.           |
| ****Embedding Model****  | `sentence-transformers/all-MiniLM-L6-v2`         | A high-performance model for converting text chunks into meaningful vector embeddings.           |
| ****Generative LLM****   | `mistralai/Mistral-7B-Instruct-v0.2` (Quantized) | A powerful instruction-tuned model for answer generation, summarization, and flashcard creation. |
| ****Speech-to-Text****   | `openai/whisper-base`                            | For accurate and fast transcription of spoken questions.                                         |
| ****Text-to-Speech****   | gTTS (Google Text-to-Speech)                     | To convert the assistant's text answers into natural-sounding speech.                            |
| ****Memory****           | SQLite                                           | A lightweight database for logging user mistakes and tracking learning progress.                 |

### Project Structure

The project follows a clean, modular architecture to separate concerns:    ai-study-assistant/
    ├── data/                 # For all raw, processed, and database files
    ├── src/                  # Main source code for the application
    │   ├── app/              # Streamlit UI code
    │   ├── features/         # High-level AI features (generator, summarizer, etc.)
    │   ├── llm/              # Centralized LLM loading logic
    │   ├── memory/           # Logic for the adaptive memory tracker
    │   ├── preprocessing/    # Scripts for data cleaning and parsing
    │   ├── rag_core/         # Core RAG components (chunker, embedder, retriever)
    │   └── voice/            # Speech-to-text and text-to-speech modules
    ├── tests/                # Automated tests for the project
    ├── config.yaml           # Central configuration file
    └── requirements.txt      # Project dependencies
====================================================

## ⚙️ Setup and Installation

# Follow these steps to set up and run the project locally.

### 1. Prerequisites

# - Python 3.10 or higher

- An NVIDIA GPU is recommended for running the local LLMs efficiently.

### 2. Clone the Repository

    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/ai-study-assistant.git
    cd ai-study-assistant
=========================

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
=======================================================================

### 4. Install Dependencies

# Install all the required Python libraries.    pip install -r requirements.txt

### 5. Hugging Face Authentication

# The generative LLM used in this project (`Mistral-7B-Instruct`) is a gated model. You need to have a Hugging Face account and be authenticated to download it.1) Visit the [Mistral-7B-Instruct-v0.2 model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 "null") and accept the terms to get access.

2) Log in to your Hugging Face account from your terminal:

       huggingface-cli login

   Paste your access token when prompted.

## 🚀 How to Use

### 1. Add Your Notes

# Place your study notes (in `.pdf` format) inside the `data/raw/` directory. You can create any sub-folder structure you like to organize them by topic (e.g., `data/raw/Machine_Learning/Supervised_Learning/`).

### 2. Build the Vector Store

# Before running the app for the first time, you need to process your notes and build the vector database. Run the following script from the root directory:    python build_vector_store.pyThis will read your raw notes, clean them, chunk them, and store them as embeddings in ChromaDB.

### 3. Launch the Application

# Once the vector store is built, you can launch the Streamlit web app:    streamlit run src/app/main.pyYour AI Study Assistant should now be running in your web browser!

## 🔧 Configuration

# All key parameters for the project are managed in the `config.yaml` file. Here you can easily change:* File paths for data and databases.

* The embedding and generative LLM models used.

* Parameters for text chunking (`chunk_size`, `chunk_overlap`).

* The number of documents to retrieve (`k`).

* The similarity threshold for the quiz engine.

## 🔮 Future Work

# This project has a solid foundation that can be extended with even more advanced features:* ****"Review My Mistakes" Quiz:**** A dedicated quiz mode that only uses questions the user has previously answered incorrectly.

* ****Spaced Repetition:**** Use the `timestamp` data in the memory tracker to implement a spaced repetition system that re-surfaces difficult concepts at optimal intervals.

* ****Multi-Modal Notes:**** Extend the data processing pipeline to handle images, diagrams, and tables within your notes.

## 📄 License

# This project is licensed under the MIT License. See the `LICENSE` file for details.

## 👤 Author

# **Ashpak Jabbar Shaikh*** [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/ashpak-shaikh-88a7372b0?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BCQ%2BNCzzzTF2LhNm6AF01yQ%3D%3D "null")

* \[GitHub]\(
