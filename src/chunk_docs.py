from pypdf import PdfReader
import os

CHUNK_SIZE = 500
DATA_PATH = 'C:\\Users\\AngadJaswal\\Projects\\rag-doc-qa\\data'

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
            # print(f"Processing document: {doc_file}")
            doc_text = load_doc(doc_file)
            # print(f"Document# {i+1} : '{doc_file}' loaded with {len(doc_text)} characters.")
            doc_chunks = chunk_pdf_text(doc_text)
            # print(f"Document# {i+1} : '{doc_file}' chunked into {len(doc_chunks)} chunks.")
            all_docs_chunks[doc_file] = doc_chunks

    # print(f"\nTotal documents processed: {len(all_docs_chunks)}")

    return all_docs_chunks

if __name__ == "__main__":

    pdf = 'HDFC-Life-Guaranteed-Wealth-Plus-101N165V13-Policy-Document.pdf'

    print(f"Loading document: {pdf}")
    pdf_text = load_doc(pdf)
    print(f"Total characters in document: {len(pdf_text)}")

    pdf_chunks = [pdf_text[i:i+CHUNK_SIZE] for i in range(0, len(pdf_text), CHUNK_SIZE)]
    print(f"Total chunks created: {len(pdf_chunks)}")

    # Print first 3 chunks for inspection
    for i, chunk in enumerate(pdf_chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:100]}... {chunk[-10:]}\n")

    # First Chunk print
    first_chunk = pdf_chunks[0]
    print(f"First chunk -> :\n{first_chunk}")

    # Chunk number middle print
    middle_index = len(pdf_chunks) // 2
    middle_chunk = pdf_chunks[middle_index] 
    print(f"\nMiddle chunk (Chunk {middle_index + 1}) -> :\n{middle_chunk}")

    # Last Chunk print
    last_chunk = pdf_chunks[-1]
    print(f"\nLast chunk -> :\n{last_chunk}")
