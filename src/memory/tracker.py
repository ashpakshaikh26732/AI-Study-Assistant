import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

import sqlite3
import datetime

def initialize_database(config):
    """
    Initializes the SQLite database and creates the 'mistakes' table if it
    does not already exist.

    This function should be called once when the application starts to ensure
    the database and table are ready for logging. It connects to the database
    file specified in the configuration.

    Args:
        config (dict): The project's configuration dictionary, which must
                     contain the path to the SQLite database file under
                     the key 'data.sqlite_database_path'.

    Side Effects:
        - Creates an SQLite database file at the specified path if it
          doesn't exist.
        - Creates the 'mistakes' table within the database.
        - Prints a confirmation message to the console.
    """
    with sqlite3.connect(config['memory']['sqlite_database_path']) as conn:
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        print('Created or found mistakes table.')

def log_mistake(topic, question, config):
    """
    Logs a single incorrect answer from a quiz into the SQLite database.

    This function records the topic of the question, the question itself,
    and the exact timestamp of when the mistake was made. This data can
    be used later to identify a user's weak areas.

    Args:
        topic (str): The topic or course associated with the question.
        question (str): The text of the question the user answered incorrectly.
        config (dict): The project's configuration dictionary, containing the
                     path to the SQLite database.

    Side Effects:
        - Inserts a new row into the 'mistakes' table in the database.
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(config['memory']['sqlite_database_path']) as conn:
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO mistakes (
            topic, question, timestamp
        ) VALUES (?, ?, ?)
        """
        mistake_data = (topic, question, formatted_datetime)
        cursor.execute(insert_query, mistake_data)
        conn.commit()
    
def get_weak_topics(config):
    """
    Queries the mistakes database to identify the topics a user struggles
    with the most.

    This function connects to the SQLite database and executes a query that:
    1.  Groups all logged mistakes by their 'topic'.
    2.  Counts the number of mistakes for each topic.
    3.  Orders the topics in descending order based on the mistake count.
    4.  Limits the results to the top N topics, as specified in the
        configuration file.

    Args:
        config (dict): The project's configuration dictionary. It must
                     contain the path to the SQLite database and the
                     limit for the number of topics to return under the
                     'memory' key.

    Returns:
        list[tuple]: A list of tuples, where each tuple contains the
                     topic name and the corresponding number of mistakes.
                     For example: [('Sequence Models', 5), ('Supervised ML', 2)]
    """
    with sqlite3.connect(config['memory']['sqlite_database_path']) as conn:
        cursor = conn.cursor()
        query = """
        SELECT topic, COUNT(*) as mistake_count
        FROM mistakes
        GROUP BY topic
        ORDER BY mistake_count DESC
        LIMIT ?
        """
        
        limit = (config['memory']['limit'],)
        cursor.execute(query, limit)
        results = cursor.fetchall()
        return results
