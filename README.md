# RAG Doc QA

Minimal Retrieval-Augmented Generation pipeline to answer questions over a small set of documents.

## About
+ RAG-DOC-QA works as a question-answering (qna) chatbot tool for some set of documents defined in data
+ The documents used herein are six dummy insurance policies available online.
+ For QnA, LLM interface is used
+ THe model is orechestrated using Python

## Structure workflow
1. **Load PDFs**- Load the downloaded pdfs using PyPdf
2. **Chunk text**- Chunk the pdf text content into small proportions
3. **Embed chunks**- Embed the Chunks in form of a Vector DB. Here, FAISS is used
4. **Retrieve relevant context**- Based on the question asked, recall the vectors as context using a pre-trained sentence transformer model
5. **Generate grounded answer**- Generate an answer from the context retrieved using a LLM model. Here, HuggingFace's zephyr-7b-beta is used

## Current limitations
+ A reletaviley small corpus of data, only six PDFs and with no heterogenity in data
+ Naive tuning is done- naive chunking, no hyperparameter tuning, no LLM tuning. Overal minimal optimization.
+ No re-ranking, so repeatetive similar contextual qna can be monotonous

## Local setup guide
1. 
