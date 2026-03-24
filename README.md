# Unigraph

Unigraph is a document retrieval and processing system designed to handle PDF ingestion, embedding, retrieval, and RAG (Retrieval-Augmented Generation) workflows. It leverages a local Chroma database for vector storage and supports advanced querying and reranking for improved information retrieval.

## Project Structure
- `app.py`: Main application entry point (Streamlit app).
- `environment.yml`: Conda environment specification.
- `chroma_db/`: Local Chroma vector database files.
- `data/`: Contains processed and raw data files.
- `retrieval/`: Core retrieval, embedding, ingestion, and RAG logic.

## How to Run

1. **Install dependencies** (recommended: use Conda):
   ```bash
   conda env create -f environment.yml
   conda activate unigraph
   ```
   Or use your preferred method to install the required packages.

2. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   Open your browser and go to the URL provided by Streamlit (usually http://localhost:8501).

---

Feel free to explore the `retrieval/` directory for the main logic and customize as needed for your use case.
