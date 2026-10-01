import streamlit as st
import requests
import subprocess
import time
import socket
import sys
import os

@st.cache_resource
def start_backend():
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if not is_port_in_use(8000):
        print("Starting FastAPI backend...")
        # Ensure 'src' is in PYTHONPATH so uvicorn can find verity.api
        env = os.environ.copy()
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env["PYTHONPATH"] = src_path
        # We start the backend and keep it running in the background.
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "verity.api:app", "--host", "127.0.0.1", "--port", "8000"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait up to 60 seconds for the port to open (models take time to load)
        for _ in range(60):
            if proc.poll() is not None:
                err = proc.stderr.read().decode()
                return f"Backend failed to start: {err}"
            if is_port_in_use(8000):
                return "OK"
            time.sleep(1)
        return "Backend did not start in 60 seconds. It might still be loading."
    else:
        return "OK"

backend_status = start_backend()

st.set_page_config(page_title="Verity-RAG Judge Demo", layout="wide")

st.title("Verity-RAG Judge Demo")
if backend_status != "OK":
    st.error(f"⚠️ Backend Status: {backend_status}")
    st.info("Check the Streamlit Cloud logs (bottom right icon -> Manage app) for more details.")

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
