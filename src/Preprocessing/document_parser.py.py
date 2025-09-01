import sys , os
import pymupdf as fitz
import pdf2image 
import pytesseract

from src.Preprocessing.text_cleaner import cleaining_fn 

repo_path = "/content/drive/MyDrive/AI-Study-Assistant"
sys.path.append(repo_path)

def extract_text_from_pdf(file_path):
    """
    Extracts text directly from a PDF file using PyMuPDF (fitz).
    This method is fast and effective for PDFs with selectable, typed text.

    Args:
        file_path (str): The full path to the PDF file.

    Returns:
        str: The extracted text from all pages of the PDF, concatenated.
             Returns an empty string if an error occurs.
    """
    extracted_text = ''
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                extracted_text += text
                extracted_text += "\n"
        return extracted_text
    except fitz.FileNotFoundError:
        print(f'PDF file {file_path} not found')
        return ""
    except Exception as e:
        print(f'An error occurred during direct text extraction: {e}')
        return ""

def ocr_pdf(file_path, config):
    """
    Performs Optical Character Recognition (OCR) on a PDF file.
    This is used for scanned documents or PDFs where text is not selectable.
    It converts each page to an image and uses Tesseract to extract text.

    Args:
        file_path (str): The full path to the PDF file.
        config (dict): The project's configuration dictionary, which should
                     contain the path for storing temporary OCR images.

    Returns:
        str: The OCR'd text from all pages of the PDF, concatenated.
             Returns an empty string if an error occurs.
    """
    ocr_text = ''
    # Get the temporary directory for OCR images from the config file.
    output_dir = config['data']['ocr_file_path']
    os.makedirs(output_dir, exist_ok=True)
    try:
        images = pdf2image.convert_from_path(file_path, dpi=300, output_folder=output_dir, fmt='jpeg')
        for image in images:
            data = pytesseract.image_to_string(image)
            ocr_text += data
            ocr_text += '\n'
        return ocr_text
    except Exception as e:
        print(f'An error occurred during OCR processing: {e}')
        return ""

def process_all_documents(config):
    """
    Orchestrates the entire document processing pipeline.
    It walks through the raw data directory, processes each PDF file by
    attempting direct text extraction and falling back to OCR, cleans the
    extracted text, and saves the result to the processed data directory,
    mirroring the original folder structure.

    Args:
        config (dict): The project's configuration dictionary containing
                     all necessary paths.

    Side Effects:
        - Creates .txt files in the `processed_path` directory.
        - Prints status messages or errors to the console.
    """
    raw_data_folder = config['data']['raw_path']
    
    print(f"Starting document processing in: {raw_data_folder}")

    for root, dirs, files in os.walk(raw_data_folder):
        if len(files) > 0:

            save_path = root.replace("raw", "processed")
            os.makedirs(save_path, exist_ok=True)

            for single_file in files:
                if single_file.endswith('.pdf'):
                    file_path = os.path.join(root, single_file)
                    print(f"Processing: {file_path}")


                    text = extract_text_from_pdf(file_path)


                    if len(text) < 200:
                        print("    --> Low text yield. Falling back to OCR...")
                        text = ocr_pdf(file_path=file_path, config=config)


                    clean_text = cleaining_fn(text)


                    path_data = root.split(os.sep)
                    path_data = path_data[2:] 
                    meta_data = {}
                    if len(path_data) > 2:
                        meta_data = {
                            'specialization': path_data[0],
                            'course': path_data[1],
                            'notes_type': path_data[2]
                        }
                    elif len(path_data) == 2:
                        meta_data = {
                            'course': path_data[0],
                            'notes_type': path_data[1]
                        }


                    file_name = f'{os.path.splitext(single_file)[0]}.txt'
                    full_file_path = os.path.join(save_path, file_name)
                    
                    try:
                        with open(full_file_path, 'w', encoding='utf-8') as doc:
                            doc.write(clean_text)
                        print(f"    --> Saved to: {full_file_path}")
                    except Exception as e:
                        print(f"    --> ERROR: Could not write file. {e}")

    print("Document processing complete.")

