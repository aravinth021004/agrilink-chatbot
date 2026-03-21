# AgriLink Chatbot

This is the backend repository for the AgriLink Chatbot, a Retrieval-Augmented Generation (RAG) based AI assistant designed to provide agricultural knowledge and support.

## Project Structure

- `main.py`: The main entry point for the application.
- `api/`: Contains the API routing and endpoints (e.g., `chat.py`).
- `rag/`: Core RAG components including the LLM chain, vector store integration, and embeddings generation.
- `knowledge/`: Local knowledge base documents (`agrilink-knowledge-base.md`) used to ground the chatbot's responses.
- `requirements.txt`: Python package dependencies.
- `Dockerfile`: Instructions for containerizing the application.

## Setup and Installation

### Prerequisites
- Python 3.8+ (or Docker)

### Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Locally
Start the server by running the main entry point:
```bash
python main.py
```

### With Docker
Build and run the containerized application:
```bash
docker build -t agrilink-chatbot .
docker run -p 8000:8000 agrilink-chatbot
```
