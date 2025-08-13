import numpy as np
import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

def grade_user_answer(user_answer, correct_answer, embedding_model, config):
    """
    Grades a user's answer by comparing its semantic meaning to the correct
    answer, rather than relying on an exact string match.

    This function performs the following steps:
    1.  It uses a pre-initialized sentence-transformer model to embed both
        the user's answer and the correct answer into numerical vectors.
    2.  It calculates the cosine similarity between these two vectors. Cosine
        similarity is a measure of how similar the directions of two vectors
        are, which corresponds to semantic similarity in this context.
    3.  It compares the resulting similarity score against a predefined
        threshold from the configuration file.

    Args:
        user_answer (str): The answer provided by the user.
        correct_answer (str): The ground truth answer for the question.
        embedding_model (langchain_huggingface.embeddings.HuggingFaceEmbeddings):
            The initialized embedding model object used to convert text to vectors.
        config (dict): The project's configuration dictionary, which must
                     contain the similarity threshold under the key
                     'features.quiz.similarity_threshold'.

    Returns:
        bool: True if the cosine similarity is greater than or equal to the
              threshold (indicating the answer is correct), False otherwise.
    """
    embeddings = embedding_model.embed_documents([user_answer, correct_answer])
    user_answer_embedding = embeddings[0]
    correct_answer_embedding = embeddings[1]

    cosine_similarity = np.dot(user_answer_embedding, correct_answer_embedding) / (np.linalg.norm(user_answer_embedding) * np.linalg.norm(correct_answer_embedding))

    if cosine_similarity >= config['features']['quiz']['similarity_threshold']:
        return True
    
    return False
