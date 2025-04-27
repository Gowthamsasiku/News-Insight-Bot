# main.py
from utils import scrape_article, chunk_text, embed_chunks, store_in_vector_db, retrieve_similar_chunks, generate_answer

def main():
    # Step 1: Get the news article
    url = input("Enter the news article URL: ")
    article_text = scrape_article(url)
    print("Article scraped successfully.")

    # Step 2: Chunk the article
    chunks = chunk_text(article_text)
    print(f"Article split into {len(chunks)} chunks.")

    # Step 3: Embed the chunks
    embeddings = embed_chunks(chunks)
    print("Chunks embedded successfully.")

    # Step 4: Store embeddings in VectorDB
    store_in_vector_db(chunks, embeddings, url)
    print("Chunks stored in Vector Database.")

    # Step 5: Ask questions
    while True:
        question = input("\nAsk a question about the article (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        similar_chunks = retrieve_similar_chunks(question)
        answer = generate_answer(similar_chunks, question)

        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
