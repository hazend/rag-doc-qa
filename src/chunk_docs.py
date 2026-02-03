from pypdf import PdfReader
import os

CHUNK_SIZE = 500
DATA_PATH = 'C:\\Users\\AngadJaswal\\Projects\\rag-doc-qa\\data' # Your 'data' path

def load_doc(DOC_NAME: str):
    pdf_path = os.path.join(DATA_PATH, DOC_NAME)
    pdf_text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages: pdf_text += page.extract_text()
    return pdf_text

def chunk_pdf_text(pdf_text: str, chunk_size: int = CHUNK_SIZE):
    return [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]

def all_doc_chunk():
    """
    Processes all PDF documents in the 'data' directory, loading their content,
    chunking the text, and storing the results in a dictionary.
    Returns:
        dict: A dictionary where the keys are the names of the processed PDF 
        documents and the values are lists of text chunks extracted from each 
        document.
    The function prints the status of each document being processed, including 
    the number of characters loaded and the number of chunks created.
    """
    all_docs_chunks = {}

    for i, doc_file in enumerate(os.listdir(DATA_PATH)):
        if doc_file.endswith('.pdf'):
            doc_text = load_doc(doc_file)
            doc_chunks = chunk_pdf_text(doc_text)
            all_docs_chunks[doc_file] = doc_chunks
    
    return all_docs_chunks
