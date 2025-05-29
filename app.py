import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Your paragraph goes here - REPLACE THIS WITH YOUR OWN PARAGRAPH
YOUR_PARAGRAPH = """
This is a sample default for now
"""

# Set up the page
st.set_page_config(page_title="Kumarbot", page_icon="🤖")
st.title("🤖 Kumarbot")
st.write("Ask me questions about the Employee Handbook!")

# Show the paragraph
with st.expander("📖 Click to see the source paragraph"):
    st.write(YOUR_PARAGRAPH)

# Initialize the embedding model (this runs once)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Split paragraph into sentences and create embeddings
@st.cache_data
def prepare_data(paragraph):
    # Split into sentences
    sentences = re.split(r'[.!?]+', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Create embeddings
    model = load_model()
    embeddings = model.encode(sentences)
    
    return sentences, embeddings

# Get OpenAI API key from user
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Enter your OpenAI API Key:", 
        type="password", 
        value=st.session_state.openai_api_key,
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    if api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Prepare the data
sentences, embeddings = prepare_data(YOUR_PARAGRAPH)

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the Handbook..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if API key is provided
    if not st.session_state.openai_client:
        with st.chat_message("assistant"):
            st.error("Please enter your OpenAI API key in the sidebar to use the chatbot.")
    else:
        # Find relevant sentences
        model = load_model()
        question_embedding = model.encode([prompt])
        
        # Calculate similarity
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top 2 most similar sentences
        top_indices = np.argsort(similarities)[-2:][::-1]
        relevant_sentences = [sentences[i] for i in top_indices if similarities[i] > 0.1]
        
        if not relevant_sentences:
            relevant_sentences = [sentences[0]]  # Fallback to first sentence
        
        # Create context
        context = " ".join(relevant_sentences)
        
        # Generate response using OpenAI
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. Answer the user's question based on this context: {context}. If the context doesn't contain enough information to answer the question, say so politely."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    answer = response.choices[0].message.content
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show which sentences were used (for debugging)
                    with st.expander("🔍 Source sentences used"):
                        for sentence in relevant_sentences:
                            st.write(f"• {sentence}")
        
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error: {str(e)}")
                st.write("Make sure your OpenAI API key is correct and you have credits available.")

# Instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.write("""
    1. Get an OpenAI API key from https://platform.openai.com/api-keys
    2. Enter your API key in the box above
    3. Ask questions about the paragraph
    4. The chatbot will find relevant sentences and answer your questions!
    """)
    
    st.header("💡 Tips")
    st.write("""
    - Ask specific questions about the content
    - Try questions like "What is the Amazon rainforest?" or "Why is deforestation a problem?"
    - The chatbot can only answer based on the paragraph provided
    """)
