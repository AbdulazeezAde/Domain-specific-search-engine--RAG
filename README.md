# Domain Specific AI Search Engine

This project is a Streamlit-based application for question-answering over your own documents using LLMs (OpenAI or Gemini) and vector search (FAISS).

## Features
- Upload PDF, DOCX, or TXT files
- Chunk and embed documents using OpenAI embeddings
- Store and search embeddings with FAISS
- Choose between OpenAI (GPT-3.5) and Gemini (Gemini 1.5 Flash) for answering questions
- View chat history

## Requirements
- Python 3.9+
- Streamlit
- langchain
- openai
- tiktoken
- faiss-cpu
- langchain-google-genai (for Gemini)
- google-generativeai (for Gemini)
- python-dotenv

## Installation
```bash
pip install streamlit langchain openai tiktoken faiss-cpu langchain-google-genai google-generativeai python-dotenv
```

## Usage
1. Run the app:
   ```bash
   streamlit run src/app.py
   ```
2. Enter your OpenAI or Gemini API key in the sidebar.
3. Upload a document and process it.
4. Ask questions about your document!

## Notes
- Gemini support requires a valid Gemini API key and the `langchain-google-genai` and `google-generativeai` packages.
- Embedding cost is estimated for OpenAI embeddings.

---

Feel free to contribute or open issues for improvements!
