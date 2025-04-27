# app.py
import streamlit as st
from utils import scrape_article, chunk_text, embed_chunks, store_in_vector_db, retrieve_similar_chunks, generate_answer

st.set_page_config(page_title="News Article QA", layout="wide")

st.title("📰 News Article Question-Answering App")

# Step 1: Input URL
url = st.text_input("Enter a news article URL:", placeholder="https://example.com/article")

if url:
    with st.spinner("Scraping the article..."):
        content = scrape_article(url)
    
    if content:
        st.success("Article scraped successfully!")
        
        with st.spinner("Chunking and embedding the article..."):
            chunks = chunk_text(content)
            embeddings = embed_chunks(chunks)
            store_in_vector_db(chunks, embeddings, url)
        
        st.success("Article processed and stored in vector DB!")
        
        st.subheader("Ask a Question about the Article")
        question = st.text_input("Your Question:", placeholder="What is the article about?")

        if question:
            with st.spinner("Searching for relevant information..."):
                context = retrieve_similar_chunks(question)
                answer = generate_answer(context, question)

            st.markdown("### Answer:")
            st.write(answer)

            st.markdown("### Relevant Context:")
            for i, ctx in enumerate(context):
                with st.expander(f"Chunk {i+1}"):
                    st.write(ctx)

    else:
        st.error("Failed to scrape the article. Please check the URL.")
