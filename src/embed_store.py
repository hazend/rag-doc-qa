from chunk_docs import all_doc_chunk
import os
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
        show_progress_bar=True
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

if __name__ == "__main__":

        
    

    # for doc_chunk in all_chunks:
    #     for chunk in doc_chunk:
    #         print(chunk[:10])  # Print first 100 characters of each chunk

    

    # print(chunk_metadata[:5])  # Print first 5 metadata entries for inspection
    # print(len(chunk_metadata))  # Print total number of chunks processed

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings for all chunks...")
    embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    print("Embeddings generated with shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)   

    print("Total vectors:", index.ntotal)

    faiss.write_index(index, "data/faiss.index")

    with open("data/all_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    with open("data/chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunk_metadata, f)

    def search(query, k=5):
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(q_emb, k)
        return [(all_chunks[i], scores[0][j]) for j, i in enumerate(indices[0])]

    query = "What is the maturity benefit of HDFC Life Click 2 Wealth Plan?"
    results = search(query, k=3)

    print(f"Top results for query: '{query}'\n")
    for i, (chunk, score) in enumerate(results):
        print(f"Result {i+1} (Score: {score:.4f}):\n{chunk}\n")