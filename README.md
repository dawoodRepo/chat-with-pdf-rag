# Chat with PDF — RAG Pipeline (Manual + LangChain)

A retrieval-augmented generation (RAG) system that answers questions from a PDF document, implemented two ways: a fully manual pipeline built from scratch, and a LangChain-based version — for comparison and learning.

## Demo
[Add a screenshot or terminal output GIF here]

## Problem
Large language models can't answer questions about documents they weren't trained on. RAG solves this by retrieving relevant chunks of a document and feeding them to the LLM as context before it answers.

## How it works
1. Load and chunk a PDF (`story.pdf`) using PyMuPDF
2. Generate embeddings for each chunk using Gemini's embedding model
3. Store embeddings in a local ChromaDB vector store
4. On a user query, embed the question and retrieve the top 3 most relevant chunks
5. Send the retrieved context + question to a Groq-hosted LLM (LLaMA 4 Scout) for the final answer

Two implementations are included:
- `manual_rag_pipeline.py` — built from scratch, no RAG framework
- `langchain_rag_pipeline.py` — same pipeline using LangChain, with a custom `GeminiEmbeddings` wrapper class since LangChain doesn't natively support Gemini embeddings

`embedding_similarity_demo.py` is a standalone script demonstrating cosine similarity between sentence embeddings — useful as a conceptual primer before the full RAG pipeline.

## Tech Stack
- Python
- Groq API (LLM inference)
- Google Gemini (embeddings)
- ChromaDB (vector store)
- PyMuPDF (PDF parsing)
- LangChain (framework-based implementation)

## Key Features
- Two RAG implementations for side-by-side comparison (manual vs. framework)
- Persistent vector storage — embeds once, reuses on future runs
- Custom LangChain-compatible embedding wrapper for Gemini

## Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file with:
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key

## What I learned
Building the manual pipeline first made it much clearer what LangChain abstracts away — particularly around embedding interfaces and retriever logic. Writing the custom `GeminiEmbeddings` class also taught me how to extend a framework's base classes to support unsupported providers.