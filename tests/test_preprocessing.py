import sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.Preprocessing.text_cleaner import cleaning_fn


def test_cleaning_function():
    """
    Tests the text cleaning functionality of the preprocessing module.

    This unit test verifies the `cleaning_fn` by checking its performance
    across five critical scenarios, ensuring it correctly normalizes various
    forms of messy whitespace.

    The test cases cover:
    1.  Collapsing multiple spaces between words.
    2.  Replacing newline and tab characters with a single space.
    3.  Stripping any leading or trailing whitespace from the text.
    4.  Handling a combination of all the above issues in a single string.
    5.  Ensuring that an already clean string remains unchanged.
    """


    # Test Case 1: Multiple Spaces
    dirty_1 = "This   string   has   too   many   spaces."
    expected_1 = "This string has too many spaces."

    # Test Case 2: Newlines and Tabs
    dirty_2 = "This\nstring\tcontains\n\nvarious whitespace characters."
    expected_2 = "This string contains various whitespace characters."

    # Test Case 3: Leading and Trailing Whitespace
    dirty_3 = "    This string has leading and trailing whitespace.    "
    expected_3 = "This string has leading and trailing whitespace."

    # Test Case 4: A Combined Mess
    dirty_4 = " \n  Here is a\t messy\n\n string with   everything. \t "
    expected_4 = "Here is a messy string with everything."

    # Test Case 5: An Already Clean String
    dirty_5 = "This is a clean string."
    expected_5 = "This is a clean string."


    assert cleaning_fn(dirty_1) == expected_1
    assert cleaning_fn(dirty_2) == expected_2
    assert cleaning_fn(dirty_3) == expected_3
    assert cleaning_fn(dirty_4) == expected_4
    assert cleaning_fn(dirty_5) == expected_5

