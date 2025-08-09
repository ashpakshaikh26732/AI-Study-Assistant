import re,sys

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

def clearning_fn(text) : 
    """
    Cleans raw extracted text by performing a series of normalization tasks.

    Args:
        text (str): The raw text extracted from a document.

    Returns:
        str: The cleaned and normalized text.
    """
    text = re.sub(r'\s+' , " " , text)
    text = text.strip() 
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return cleaned_text 