# utils.py
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import chromadb

# import nltk

# for resource in ['punkt', 'punkt_tab']:
#     try:
#         nltk.data.find(f'tokenizers/{resource}')
#     except LookupError:
#         nltk.download(resource)

import nltk
import os

def ensure_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Call it safely once
ensure_nltk_punkt()


# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("news_articles")

def scrape_article(url):
    """Scrape text content from a news article URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return content

def chunk_text(text, chunk_size=5):
    """Split text into smaller chunks"""
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def embed_chunks(chunks):
    """Generate embeddings for each chunk"""
    embeddings = embed_model.encode(chunks)
    return embeddings

def store_in_vector_db(chunks, embeddings, url):
    """Store chunks and their embeddings in the vector database"""
    # Make sure embeddings are lists
    embeddings = [embedding.tolist() for embedding in embeddings]

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[{"url": url, "chunk_id": i}],
            ids=[f"{url}_{i}"]
        )

def retrieve_similar_chunks(question, top_k=3):
    """Retrieve top-k similar chunks for a given question"""
    q_embedding = embed_model.encode([question])
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k
    )
    return [doc for doc in results['documents'][0]]

def generate_answer(context, question):
    """Generate answer from context using a small Huggingface model"""
    from transformers import pipeline

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

    context_text = "\n".join(context)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    
    output = qa_pipeline(prompt, max_length=200)
    return output[0]['generated_text']
