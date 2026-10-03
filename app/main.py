from fastapi import FastAPI
from app.config import settings
from app.parser.pdf_parser import PDFParser
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever
from app.rag.rag_service import RAGService

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

@app.get("/test-embedding")
def test_embedding():

    # Parse PDF
    parser = PDFParser(r"uploads\SSC-GA-GS-Capsule-new.pdf")
    document = parser.parse()

    # Create chunks
    chunker = TextChunker()
    chunks = chunker.chunk(document)

    # Create embedding service
    embedding_service = EmbeddingService()

    # Embed first chunk only
    embedding = embedding_service.embed_text(
        chunks[0].text
    )

    return {
        "chunk_id": chunks[0].chunk_id,
        "token_count": chunks[0].token_count,
        "embedding_dimensions": len(embedding),
        "embedding_preview": embedding[:5]
    }

@app.post("/ingest")
def ingest_pdf():

    # 1. Parse PDF
    parser = PDFParser(
        r"uploads\SSC-GA-GS-Capsule-new.pdf"
    )

    document = parser.parse()

    # 2. For practice, use only first 10 pages
    document.pages = document.pages[:10]

    # 3. Create chunks
    chunker = TextChunker()

    chunks = chunker.chunk(document)

    # 4. Generate embeddings
    embedding_service = EmbeddingService()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedding_service.embed_texts(texts)

    # 5. Store in ChromaDB
    chroma_store = ChromaStore()

    chroma_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "status": "success",
        "document": document.name,
        "pages_processed": len(document.pages),
        "chunks_created": len(chunks),
        "embeddings_created": len(embeddings),
        "stored_in": "ChromaDB",
    }


@app.get("/search")
def search(query: str, top_k: int = 5):

    embedding_service = EmbeddingService()

    query_embedding = embedding_service.embed_text(query)

    retriever = Retriever()

    results = retriever.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return {
        "query": query,
        "results": results,
    }

@app.get("/ask")
def ask_question(
    question: str,
    session_id: str,
    top_k: int = 5,
):

    rag_service = RAGService()

    result = rag_service.answer(
        question=question,
        session_id=session_id,
        top_k=top_k,
    )

    return result