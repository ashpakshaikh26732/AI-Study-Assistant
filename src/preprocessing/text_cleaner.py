import re


def cleaning_fn(text: str) -> str:
    """Normalize raw extracted text while keeping its line/paragraph structure.

    Paragraph breaks matter: the chunker prefers to split on blank lines, so
    flattening everything into one line (as the first version did) made chunks
    cut mid-sentence. This function therefore:

    * drops BOMs and unifies line endings,
    * re-joins words hyphenated across a line break ("classi-\\nfication"),
    * collapses runs of spaces/tabs to one space,
    * trims spaces around line breaks and caps blank lines at one.

    Args:
        text: Raw text from a PDF, OCR pass or text file.

    Returns:
        The cleaned text.
    """
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    # Only join when the next fragment starts lowercase, so "well-\nKnown" style
    # proper nouns and bullet lists are left alone.
    text = re.sub(r"(\w+)-[ \t]*\n[ \t]*([a-z]\w*)", r"\1\2", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
