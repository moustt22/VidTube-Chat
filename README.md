# 🎥 VidTube Chat

VidTube Chat is a **YouTube Video Question-Answering application** built using **Retrieval-Augmented Generation (RAG)**. It allows users to chat with YouTube videos by asking natural language questions and receiving accurate, context-aware answers based on the video transcript.

Each YouTube video gets its **own vector database**, enabling isolated, accurate retrieval and multi-turn conversations per video.

---

## 🚀 Features

* 🔍 Ask questions about any YouTube video using its video ID
* 🧠 RAG pipeline with transcript chunking + embeddings
* 📦 Separate ChromaDB vector store per video
* 💬 Multi-turn conversational memory per video
* 📊 Optional retrieval evaluation (relevance scoring)
* 🖥️ Clean chat-style UI using Streamlit

---

## 🧠 How It Works

1. User enters a **YouTube video ID**
2. Video transcript is fetched
3. Transcript is:

   * Chunked using time-based + text splitting
   * Embedded using OpenAI embeddings
   * Stored in a **video-specific ChromaDB folder**
4. User asks a question
5. Relevant chunks are retrieved from ChromaDB
6. An LLM generates an answer grounded only in retrieved context
7. Chat history is preserved per video

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – UI
* **LangChain** – RAG orchestration
* **ChromaDB** – Vector storage
* **OpenAI / OpenRouter** – Embeddings & LLMs
* **YouTube Transcript API** (or equivalent)

---

## 📂 Project Structure

```
VidTube-Chat/
│
├── main.py                # Streamlit app
├── ingestion.py           # Chunking + embedding logic
├── retrieval.py           # Query answering logic
├── get_transcript.py      # Fetch YouTube transcript
├── chromadb/              # Per-video vector databases
├── .env                   # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run main.py
```

Then open the browser link shown in the terminal.

---

## 🧪 Example Usage

1. Enter YouTube video ID
2. Wait for embeddings to be created (first run only)
3. Ask questions like:

   * "What is this video about?"
   * "Explain the main idea in simple terms"
   * "What did the speaker say about X?"

---

## 📈 Future Improvements

* Timestamp-based answer highlighting
* Source citation per answer
* Automatic transcript language detection
* UI enhancements
* Deployment (Docker / Cloud)

---

## 📜 License

This project is for educational and research purposes.

---

## 👤 Author

**Moustafa Mohamed**

AI & Data Science Student – Cairo University

---

⭐ If you like this project, consider giving it a star on GitHub!
