# 🎙️ Podcast AI Assistant & Content Generator

An intelligent application that transforms long-form podcast transcripts into a queryable knowledge base. Ask questions, get summaries, and generate new content in seconds. This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline, from data processing to a user-friendly web interface.

<!-- TODO: Add a GIF of your app in action! It's the best way to show what it does.
     You can use a free tool like ScreenToGif or Kap. -->
![App Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)

**[➡️ View Live Demo Here]**(https://your-streamlit-app-link.streamlit.app/) <!-- TODO: Replace with your actual deployment link -->

---

## ✨ Features

-   **Conversational Q&A:** Ask questions in natural language and get concise, context-aware answers sourced directly from the podcast transcripts.
-   **On-the-Fly Summarization:** Generate summaries of specific topics or entire episodes.
-   **Content Extraction:** Pull out key takeaways, mentioned names, or companies in a structured format.
-   **Content Repurposing:** Transform podcast discussions into new formats like Twitter threads or blog post outlines.
-   **Source Citing:** Each response includes the specific text chunks from the transcripts that were used to generate the answer, ensuring transparency and trust.

---

## 🛠️ Tech Stack & Architecture

This project is built on a modern, modular stack designed for building LLM applications.

-   **LLM Framework:** [LlamaIndex](https://www.llamaindex.ai/) for orchestrating the RAG pipeline.
-   **LLM & Embeddings:** [Google Gemini API](https://ai.google.dev/) (`gemini-1.5-flash-latest` for generation, `embedding-001` for embeddings).
-   **Web Framework:** [Streamlit](https://streamlit.io/) for creating the interactive user interface.
-   **Language:** Python 3.10+
-   **Environment:** `python-dotenv` for managing API keys.

### Architecture Overview

The application follows a classic RAG pattern:

1.  **Offline Indexing (`engine.py`):**
    -   Podcast transcripts are loaded from the `/data` directory.
    -   Text is split into smaller, manageable chunks.
    -   Each chunk is converted into a vector embedding using Gemini's embedding model.
    -   The embeddings and corresponding text are stored in a local vector index using LlamaIndex's `StorageContext`.

2.  **Real-time Querying (`app.py`):**
    -   A user submits a query through the Streamlit interface.
    -   The query is converted into an embedding using the same model.
    -   The vector index is searched to find the most semantically similar text chunks (the "context").
    -   The context and the original query are passed to the Gemini Pro model in a carefully crafted prompt.
    -   The LLM generates a response based *only* on the provided context, which is then streamed back to the user.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.10 or higher
-   A [Google Gemini API Key](https://ai.google.dev/tutorials/setup)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/podcast-qa.git
cd podcast-qa
```

### 2. Set Up a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Gemini API key to this file:

    ```env
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

### 5. Prepare Your Data

-   Place your podcast transcript `.txt` files inside the `/data` directory.

### 6. Build the Vector Index

Run the indexing script. This only needs to be done once, or whenever you add new documents to the `/data` folder.

```bash
python engine.py
```

This will create a `storage` directory containing your vector index.

### 7. Run the Streamlit App

You're all set! Launch the application with:

```bash
streamlit run app.py
```

Your web browser should automatically open with the app running.

---

## 💡 Future Improvements

-   [ ] Implement conversation memory to allow for follow-up questions.
-   [ ] Add support for more data sources (e.g., YouTube video transcripts, RSS feeds).
-   [ ] Explore more advanced retrieval strategies like hybrid search.
-   [ ] Containerize the application with Docker for easier deployment.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.