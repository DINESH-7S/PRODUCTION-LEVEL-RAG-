import logging
import os
import uuid
import shutil
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas import QueryRequest, QueryResponse, TaskStatusResponse
from parser import DocumentParser, EmbeddingGenerator, MODEL_DIMENSIONS, DEFAULT_ENCODER
from vector_db import QdrantEngine
from pipeline import RAGCoordinator

load_dotenv()

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGBackend")

app = FastAPI(title="Insight RAG API Service")

# Allow CORS for ease of integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory background task status store
tasks_status: Dict[str, Dict[str, Any]] = {}
# Concurrency lock to prevent thread/GPU starvation during heavy embedding generation
processing_semaphore = asyncio.Semaphore(2)

async def background_ingest_worker(
    task_id: str,
    file_path: str,
    doc_id: str,
    embedding_model: str
):
    """Background task processing segment extraction and vector database ingestion."""
    async with processing_semaphore:
        try:
            tasks_status[task_id].update({
                "status": "extracting",
                "progress": 25.0,
                "message": "Extracting text passages from document..."
            })
            
            # Parse PDF
            segments = DocumentParser.parse_pdf(file_path, doc_id=doc_id)
            
            if not segments:
                tasks_status[task_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "message": "No text content found in document.",
                    "processed_count": 0
                })
                return

            tasks_status[task_id].update({
                "status": "vectorizing",
                "progress": 60.0,
                "message": f"Generating vector embeddings using {embedding_model}..."
            })

            # Embed and Index
            texts = [s.content for s in segments]
            # Since generating embeddings is a blocking call (running on CPU/GPU), run in a separate executor to prevent blocking the event loop
            loop = asyncio.get_running_loop()
            vectors = await loop.run_in_executor(None, lambda: EmbeddingGenerator.embed_texts(texts, model_name=embedding_model))

            dim = MODEL_DIMENSIONS.get(embedding_model, 384)
            db_engine = QdrantEngine(vector_dim=dim)
            db_engine.store_segments(segments, vectors)

            tasks_status[task_id].update({
                "status": "completed",
                "progress": 100.0,
                "message": f"Successfully ingested {len(segments)} segments.",
                "processed_count": len(segments)
            })
        except Exception as e:
            logger.error(f"Failed to ingest document task {task_id}: {str(e)}")
            tasks_status[task_id].update({
                "status": "failed",
                "progress": 100.0,
                "message": "Internal processing failed.",
                "error": str(e)
            })
        finally:
            # Remove temporary uploaded file
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass

@app.post("/api/documents/ingest", response_model=TaskStatusResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form(DEFAULT_ENCODER)
):
    """Endpoint to upload a PDF file and trigger background indexing pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if embedding_model not in MODEL_DIMENSIONS:
        embedding_model = DEFAULT_ENCODER

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    task_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (".", "-", "_"))
    temp_path = os.path.join("uploads", f"{task_id}-{safe_filename}")

    # Write file to disk temporarily
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Enqueued for ingestion...",
        "processed_count": 0,
        "error": None
    }

    # Queue background task
    background_tasks.add_task(
        background_ingest_worker,
        task_id=task_id,
        file_path=temp_path,
        doc_id=file.filename,
        embedding_model=embedding_model
    )

    return tasks_status[task_id]

@app.get("/api/documents/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Retrieve current processing progress of an ingestion task."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found.")
    return tasks_status[task_id]

@app.post("/api/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """Execute synchronous RAG question answering pipeline."""
    coordinator = RAGCoordinator()
    response = coordinator.process_query(
        question=request.question,
        strategy=request.strategy,
        top_k=request.top_k,
        embedding_model=request.embedding_model,
        chat_history=request.chat_history,
        enable_crag=request.enable_crag,
        answer_length=request.answer_length
    )
    return response

@app.get("/api/documents", response_model=List[str])
async def list_ingested_documents():
    """Retrieve list of unique document sources stored in the system."""
    db_engine = QdrantEngine()
    return db_engine.list_documents()

@app.delete("/api/database/wipe")
async def wipe_database(embedding_model: str = DEFAULT_ENCODER):
    """Clears uploaded logs, deletes temporary files, and wipes vector store."""
    try:
        # Wipe Qdrant indexes
        dim = MODEL_DIMENSIONS.get(embedding_model, 384)
        db_engine = QdrantEngine(vector_dim=dim)
        db_engine.collection_name = "docs"
        db_engine.reset_collection()
        
        # Clean local storage
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                filepath = os.path.join(uploads_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception:
                    pass
                    
        # Reset background statuses
        tasks_status.clear()
        
        return {"status": "success", "message": "Database and temporary storage successfully cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")
