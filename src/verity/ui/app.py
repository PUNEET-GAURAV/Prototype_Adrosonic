import os
import socket
import subprocess
import sys
import time

import requests
import streamlit as st


@st.cache_resource
def start_backend():
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if not is_port_in_use(8000):
        print("Starting FastAPI backend...")
        # Ensure 'src' is in PYTHONPATH so uvicorn can find verity.api
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        # We start the backend and keep it running in the background.
        subprocess.Popen([sys.executable, "-m", "uvicorn", "verity.api:app", "--host", "127.0.0.1", "--port", "8000"], env=env)
        # Wait up to 30 seconds for the port to open
        for _ in range(30):
            if is_port_in_use(8000):
                print("FastAPI backend started successfully.")
                break
            time.sleep(1)
        else:
            print("Warning: FastAPI backend did not start in time.")
    else:
        print("FastAPI backend is already running on port 8000.")
    return True

start_backend()

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
        except requests.RequestException as e:
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
        except requests.RequestException as e:
            st.error(f"Failed to connect: {e}")
