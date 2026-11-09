# Own Vector Database and RAG System

## About the Project

This project is a Python-based Vector Database system designed to store, manage, and search high-dimensional vectors efficiently. It implements multiple vector search algorithms and distance metrics to find the most relevant results based on similarity.

The project also integrates **Ollama** to support text embeddings and Retrieval-Augmented Generation (RAG). Users can insert documents, search for relevant information, and generate AI-powered answers using the stored documents.

## Features

- **Multiple Search Algorithms:** Brute Force, KD-Tree, and HNSW.
- **Distance Metrics:** Euclidean Distance, Cosine Distance, and Manhattan Distance.
- **Vector Management:** Insert, search, and delete vector records.
- **Performance Benchmarking:** Compare search execution times across algorithms.
- **Document Processing:** Split large documents into smaller text chunks.
- **AI Embeddings:** Generate document embeddings using Ollama.
- **RAG Pipeline:** Retrieve relevant document chunks and generate answers using an AI language model.
- **REST API:** Access database functionality through HTTP endpoints.

## Technologies Used

- Python
- Ollama
- HNSW
- KD-Tree
- Vector Search
- REST API
- Retrieval-Augmented Generation (RAG)

## AI Models

This project uses the following Ollama models:

- `nomic-embed-text` – For generating text embeddings.
- `llama3.2` – For AI-powered answer generation.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Install Python

Install Python 3.9 or later.

### 3. Install Ollama

Download Ollama from:

[https://ollama.com/download](https://ollama.com/download)

### 4. Download Required Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 5. Run the Project

```bash
python vectordb_server.py
```

### 6. Open in Browser

```text
http://localhost:8080
```

Note: The browser interface requires an `index.html` file in the project folder. The Python server itself uses Python's standard library and does not require third-party Python packages.

## API Endpoints

| Method | Endpoint      | Purpose                        |
| ------ | ------------- | ------------------------------ |
| GET    | `/status`     | Check server and Ollama status |
| GET    | `/stats`      | View database statistics       |
| GET    | `/items`      | View stored vectors            |
| GET    | `/search`     | Search similar vectors         |
| GET    | `/benchmark`  | Compare search performance     |
| GET    | `/hnsw-info`  | View HNSW information          |
| POST   | `/insert`     | Insert a vector                |
| POST   | `/doc/insert` | Insert and process a document  |
| POST   | `/doc/search` | Search relevant documents      |
| POST   | `/doc/ask`    | Ask questions using RAG        |
| GET    | `/doc/list`   | List stored documents          |

## Project Objectives

1. Understand how vector databases store and retrieve vector embeddings.
2. Implement and compare different nearest-neighbor search algorithms.
3. Explore different distance metrics for similarity search.
4. Understand how AI embeddings represent textual information.
5. Implement a basic RAG workflow using local AI models.

## Future Improvements

- Add a modern and responsive frontend.
- Support persistent database storage.
- Improve search performance for larger datasets.
- Add authentication and API security.
- Support additional embedding and language models.

## Author

**Subhajit Sahoo**

B.Tech CSE (AI & ML)

## License

This project is intended for educational and learning purposes.
