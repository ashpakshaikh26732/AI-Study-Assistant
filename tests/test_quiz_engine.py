import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.features.quiz_engine import grade_user_answer



def test_grade_user_answer():
    """
    Tests the semantic answer grading functionality of the quiz engine.

    This unit test verifies the `grade_user_answer` function by checking its
    performance across three critical scenarios:
    1.  **Identical Sentences:** Ensures the function returns True for identical answers.
    2.  **Semantically Similar Sentences:** Ensures the function correctly
        identifies answers that have the same meaning but different wording,
        returning True.
    3.  **Dissimilar Sentences:** Ensures the function correctly identifies
        answers with different meanings, returning False.

    The test uses a real embedding model to perform the similarity check,
    making it an effective integration test for this core feature.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    

    mock_config = {
        "features": {
            "quiz": {
                "similarity_threshold": 0.85
            }
        }
    }


    similar_pair = ("A GRU has two gates", "There are two gates in a GRU")
    dissimilar_pair = ("A GRU has two gates", "The sky is blue")
    identical_pair = ("An RNN processes sequences", "An RNN processes sequences")

    result_similar = grade_user_answer(
        similar_pair[0], similar_pair[1], embedding_model, mock_config
    )
    assert result_similar is True


    result_dissimilar = grade_user_answer(
        dissimilar_pair[0], dissimilar_pair[1], embedding_model, mock_config
    )
    assert result_dissimilar is False

    result_identical = grade_user_answer(
        identical_pair[0], identical_pair[1], embedding_model, mock_config
    )
    assert result_identical is True
