# PDF-Based AI Chatbot

An AI/ML-powered chatbot that allows users to upload PDF documents and ask questions based on the content of those documents.

---

##  Features

- Upload PDF documents
- Chat with an AI about the document content
- Embedding-powered NLP with Sentence Transformers
- Lightweight UI via Flask
- Easy deployment with Docker
##  NLP approach
🧠 NLP Approach
Embedding Model: all-MiniLM-L6-v2 from Sentence Transformers

Similarity Matching: NearestNeighbors from scikit-learn (instead of FAISS)

Text Chunking: The document is split into 500-character text blocks

Query Flow:

User query → embedded

Nearest text chunk retrieved

Response returned from matching chunk

---

##  Setup Instructions

### 1. Clone the Repo

```bash
git https://github.com/HafsaWaheed54/pdf-chatbot-new.git
cd chatbot.py

python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python main.py
