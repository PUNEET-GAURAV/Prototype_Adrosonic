# This file acts as a root entrypoint for platforms like Hugging Face Spaces
import sys
import os

# Add src to pythonpath so verity can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import the actual Streamlit app
import verity.ui.app
