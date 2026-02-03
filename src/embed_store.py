from chunk_docs import all_doc_chunk
from sentence_transformers import SentenceTransformer
import faiss
import pickle

pdf_data = all_doc_chunk()
all_chunks = list(pdf_data.values())
chunk_metadata = []
for pdf_name, chunks in pdf_data.items():
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
            "source": pdf_name,
            "chunk_id": i
        })

# Function to perform search on the FAISS index
def search_with_metadata(query, index, k=5):
    """
    Search for the most similar chunks to a given query using semantic similarity.
    
    This function encodes the input query into a vector embedding using a pre-trained
    sentence transformer model, then searches a FAISS index to find the k most similar
    chunks based on their embeddings.
    
    Args:
        query (str): The search query string to find similar chunks for.
        k (int, optional): The number of most similar chunks to return. Defaults to 5.
    
    Returns:
        list: A list of dictionaries, each containing:
            - "chunk" (str): The text content of the matching chunk.
            - "metadata" (dict): Associated metadata for the chunk.
            - "distance" (float): The semantic distance/similarity score between
              the query and chunk (lower values indicate higher similarity).
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        result = {
            "chunk": all_chunks[idx],
            "metadata": chunk_metadata[idx],
            "distance": float(dist)
        }
        results.append(result)
    return results
