import streamlit as st
import requests

st.set_page_config(page_title="Verity-RAG Judge Demo", layout="wide")

st.title("Verity-RAG Judge Demo")
st.write("Compare Dense and Hybrid retrieval side-by-side.")

query = st.text_input("Enter a search query:", value="What is the capital of France?")

k = st.slider("Number of results (k)", min_value=1, max_value=20, value=5)

if st.button("Search"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dense Search")
        try:
            r_dense = requests.post("http://localhost:8000/search", json={"query": query, "mode": "dense", "k": k}, timeout=10)
            if r_dense.status_code == 200:
                hits = r_dense.json().get("hits", [])
                for i, h in enumerate(hits):
                    st.markdown(f"**{i+1}. Score: {h['score']:.4f}**")
                    st.write(h["text"])
            else:
                st.error(f"Error: {r_dense.text}")
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            
    with col2:
        st.subheader("Hybrid Search")
        try:
            r_hybrid = requests.post("http://localhost:8000/search", json={"query": query, "mode": "hybrid", "k": k}, timeout=10)
            if r_hybrid.status_code == 200:
                hits = r_hybrid.json().get("hits", [])
                for i, h in enumerate(hits):
                    st.markdown(f"**{i+1}. Score: {h['score']:.4f}**")
                    st.write(h["text"])
            else:
                st.error(f"Error: {r_hybrid.text}")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

