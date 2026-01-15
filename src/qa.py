import os
import faiss
import pickle
from embed_store import search_with_metadata
from huggingface_hub import InferenceClient

# Loading FAISS index, chunks and metadata from disk
print("Loading FAISS index, chunks and metadata from disk...")
index = faiss.read_index("data/faiss.index")
with open("data/all_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("data/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f'''Vectors loaded: {index.ntotal}, Chunks loaded: {len(chunks)}, Metadata entries loaded: {len(metadata)}''')

if __name__ == "__main__":

    question = "What is the maturity benefit of HDFC Life Click 2 Wealth Plan?"
    print(f"\nQuestion: {question}")

    results = search_with_metadata(question, k=5)
    chunks = [res["chunk"] for res in results]
    print(f"\nRetrieved {len(chunks)} relevant chunks")

    prompt = f'''
Use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context: {"\n".join(chunks)}

Question: {question}

Answer:
'''
    print("\n...Sending request to Hugging Face Inference API, model zephyr-7b-beta...")
    HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN=xxxx
    client = InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta"
    )


    response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=300,
    temperature=0.1
    )

    answer = response.choices[0].message.content

    print(f"\n\nAnswer:\n{answer}")