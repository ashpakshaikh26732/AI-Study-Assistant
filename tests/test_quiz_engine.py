import pytest

from src.features.quiz_engine import grade_answer, grade_user_answer, similarity_score

CONFIG = {"features": {"quiz": {"similarity_threshold": 0.85, "close_margin": 0.15}}}


class TestGradingLogic:
    """Uses the fake bag-of-words embedder, so no model download is needed."""

    def test_identical_answers_score_one(self, fake_embeddings):
        assert similarity_score("an rnn processes sequences", "an rnn processes sequences", fake_embeddings) == pytest.approx(1.0)

    def test_word_order_does_not_matter_for_meaning_match(self, fake_embeddings):
        assert grade_user_answer("gru has two gates", "two gates has gru", fake_embeddings, CONFIG) is True

    def test_unrelated_answer_fails(self, fake_embeddings):
        assert grade_user_answer("the sky is blue", "a gru has two gates", fake_embeddings, CONFIG) is False

    def test_blank_answer_scores_zero(self, fake_embeddings):
        assert similarity_score("", "anything", fake_embeddings) == 0.0
        assert similarity_score("   ", "anything", fake_embeddings) == 0.0

    def test_correct_and_incorrect_verdicts(self, fake_embeddings):
        verdict, score = grade_answer("a gru has two gates", "a gru has two gates", fake_embeddings, CONFIG)
        assert (verdict, round(score, 3)) == ("correct", 1.0)
        assert grade_answer("completely different topic", "a gru has two gates", fake_embeddings, CONFIG)[0] == "incorrect"

    def test_close_band_boundaries(self):
        class Fixed:  # returns vectors with a chosen cosine similarity
            def __init__(self, cos):
                self.cos = cos

            def embed_documents(self, texts):
                import math
                return [[1.0, 0.0], [self.cos, math.sqrt(1 - self.cos ** 2)]]

        assert grade_answer("a", "b", Fixed(0.90), CONFIG)[0] == "correct"
        assert grade_answer("a", "b", Fixed(0.75), CONFIG)[0] == "close"
        assert grade_answer("a", "b", Fixed(0.60), CONFIG)[0] == "incorrect"


@pytest.mark.model
def test_semantic_grading_with_the_real_embedding_model():
    """Same-meaning answers pass, unrelated ones fail (needs the MiniLM model)."""
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  encode_kwargs={"normalize_embeddings": True})
    assert grade_user_answer("A GRU has two gates", "There are two gates in a GRU", model, CONFIG) is True
    assert grade_user_answer("A GRU has two gates", "The sky is blue", model, CONFIG) is False
    assert grade_user_answer("An RNN processes sequences", "An RNN processes sequences", model, CONFIG) is True
