import faiss
import pickle
from huggingface_hub import InferenceClient
from embed_store import search_with_metadata
import time


# -----------------------------
# Config
# -----------------------------
FAISS_INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/all_chunks.pkl"
METADATA_PATH = "data/chunk_metadata.pkl"
QUESTION_FILE = "C:\\Users\\AngadJaswal\\Projects\\rag-doc-qa\\src\\question.txt"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
TOP_K = 5


# -----------------------------
# Load persisted artifacts
# -----------------------------
def load_vector_store(index_flg=True, chunk_metadata=False):
    if index_flg or chunk_metadata:
        print("Loading FAISS index and chunk data...")

    if index_flg: index = faiss.read_index(FAISS_INDEX_PATH)

    if chunk_metadata:
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

    if index_flg: 
        print(f"Vectors loaded: {index.ntotal}")
    if chunk_metadata:
        print(f"Chunks loaded: {len(chunks)}")
        print(f"Metadata entries loaded: {len(metadata)}")

    if chunk_metadata and index_flg: return index, chunks, metadata
    elif index_flg: return index
    else: return chunks, metadata


# -----------------------------
# Read question from file
# -----------------------------
def read_question(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        question = f.read().strip()

    if not question:
        raise ValueError("Question file is empty")

    return question


# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(chunks, question):
    context = "\n".join(chunks)

    return f"""
Use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""".strip()


# -----------------------------
# LLM call (HF chat)
# -----------------------------
def generate_answer(prompt, temperature=1, max_tokens=1000):
    client = InferenceClient(model=MODEL_NAME)

    response = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content


# -----------------------------
# Main QnA pipeline
# -----------------------------
def run_qna():

    start_time = time.time()

    # Load data (index is loaded only to ensure availability)
    index = load_vector_store()

    # Read question
    question = read_question(QUESTION_FILE)
    print(f"\nQuestion:\n{question}")

    # Retrieve
    results = search_with_metadata(question, index, k=TOP_K)
    chunks = [res["chunk"] for res in results]

    print(f"\nRetrieved {len(chunks)} chunks")

    # Build prompt
    prompt = build_prompt(chunks, question)

    # Generate answer
    print("\nQuerying LLM...")
    answer = generate_answer(prompt, temperature=0.1)

    print("\nAnswer:\n")
    print(answer)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run_qna()
