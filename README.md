# AI Study Assistant (Local RAG Application)

This project is a fully-functional, conversational AI designed to act as an expert tutor on Artificial Intelligence. It leverages a Retrieval-Augmented Generation (RAG) architecture to answer questions based on a curated knowledge base of technical documents, running entirely on a local machine using Ollama and Docker.

## Features

- **Private & Local:** Runs completely on your local machine, ensuring data privacy and zero API costs.
- **Context-Aware Answers:** Utilizes a RAG pipeline to retrieve relevant information from a local vector database (ChromaDB), dramatically reducing the risk of model hallucination.
- **Optimized for Speed:** Configured to use lightweight models like `gemma:2b` for a responsive experience on consumer hardware (CPU).
- **Fully Containerized:** The entire application stack, including the Ollama server, is managed with Docker for one-command setup and guaranteed reproducibility.
- **Modern Web API:** The backend is built with FastAPI, providing a robust and well-documented API for any potential frontend application.

## Tech Stack

- **Backend:** Python, FastAPI
- **AI/ML:** LangChain, Ollama
- **Vector Database:** ChromaDB
- **Containerization:** Docker
- **Core Models:**
    - **Generation:** `gemma:2b` (lightweight) / `llama3` (high-quality)
    - **Embeddings:** `nomic-embed-text`

## Project Architecture

1.  **Data Ingestion:** A Python script (`ingest.py`) processes PDF documents, splits them into chunks, and uses the `nomic-embed-text` model to create vector embeddings. These are stored in a local ChromaDB database.
2.  **Multi-Container Setup:** The application runs in two Docker containers on a shared network (`ai_network`):
    - `ollama-server`: Serves the LLMs (`gemma:2b`, etc.).
    - `ai-tutor-container`: Runs the FastAPI application.
3.  **RAG Pipeline:** When a user asks a question via the API:
    - The `ai-tutor-container` creates an embedding of the question.
    - It queries ChromaDB to find the most relevant document chunks.
    - The question and the retrieved context are passed to `gemma:2b`.
    - The final, context-aware answer is returned to the user.

## How to Run This Project

### Prerequisites

-   [Docker](https://www.docker.com/products/docker-desktop/) must be installed and running.
-   Sufficient RAM (8GB+ recommended for `gemma:2b`, 16GB+ for `llama3`).

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-tutor-app.git
cd ai-tutor-app