from fastapi import FastAPI
from app.config import settings
from app.parser.pdf_parser import PDFParser
from app.chunking.text_chunker import TextChunker

app = FastAPI(
    title="PDF RAG Agent",
    description="Chat with Pdfs using OpenAI + ChromaDB",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Welcome to PDF RAG Agent"
    }

@app.get("/health")
def health():
    return {
        "status":"Running",
        "application": "PDF RAG Agent",
        "embedding_model": settings.EMBEDDING_MODEL,
        "chat_model": settings.CHAT_MODEL
    }

@app.get("/test-parser")
def test_parser():

    parser = PDFParser(r"uploads\SSC-GA-GS-Capsule-new.pdf")

    document = parser.parse()

    return {
        "document": document.name,
        "pages": len(document.pages),
        "first_page": document.pages[0].page_number,
        "characters": document.pages[0].char_count,
        "text": document.pages[0].text[:200]
    }

@app.get("/test-chunks")
def test_chunks():
    parser = PDFParser(r"uploads\SSC-GA-GS-Capsule-new.pdf")

    document = parser.parse()

    chunker = TextChunker()

    chunks = chunker.chunk(document)

    return {
        "document": document.name,
        "total_pages": len(document.pages),
        "total_chunks": len(chunks),
        "first_chunk": {
            "chunk_id": chunks[0].chunk_id,
            "start_page": chunks[0].start_page,
            "end_page": chunks[0].end_page,
            "token_count": chunks[0].token_count,
            "text": chunks[0].text[:1000],
        },
        "second_chunk": {
            "chunk_id": chunks[1].chunk_id,
            "start_page": chunks[1].start_page,
            "end_page": chunks[1].end_page,
            "token_count": chunks[1].token_count,
            "text": chunks[1].text[:1000],
        },
    }

