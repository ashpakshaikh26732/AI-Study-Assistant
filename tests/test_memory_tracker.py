import sys
import os 

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.memory.tracker import initialize_database , log_mistake , get_weak_topics

def test_tracker_functions():
    """
    Tests the end-to-end functionality of the memory tracker database.

    This unit test verifies the entire workflow of the memory tracker by:
    1.  **Arranging** an isolated, temporary file-based SQLite database to ensure
        all operations in the test share the same connection context.
    2.  **Initializing** the 'mistakes' table in the temporary database.
    3.  **Acting** by logging several simulated user mistakes for different topics.
    4.  **Asserting** that the `get_weak_topics` function correctly queries the
        database to count, group, and sort the mistakes.
    5.  **Cleaning up** by removing the temporary database file.
    """

    temp_db_path = "test_memory.db"
    mock_config = {
        "memory": {
            "sqlite_database_path": temp_db_path,
            "limit": 3
        }
    }

    try:

        initialize_database(mock_config)


        log_mistake("Topic A", "Question about RNNs", mock_config)
        log_mistake("Topic B", "Question about Vectors", mock_config)
        log_mistake("Topic A", "Another question about RNNs", mock_config)


        weak_topics = get_weak_topics(mock_config)

        assert weak_topics == [('Topic A', 2), ('Topic B', 1)]

    finally:

        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)




