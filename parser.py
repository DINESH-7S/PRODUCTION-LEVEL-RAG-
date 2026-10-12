import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schemas import TextSegment

load_dotenv()

# Embedding model constants
ENCODER_MINILM = "all-MiniLM-L6-v2"
ENCODER_BGE = "BAAI/bge-small-en-v1.5"
DEFAULT_ENCODER = ENCODER_MINILM

MODEL_DIMENSIONS: Dict[str, int] = {
    ENCODER_MINILM: 384,
    ENCODER_BGE: 384,
}

_model_registry = {}

class DocumentParser:
    """Handles text extraction and chunk segmentation for PDF files."""
    
    @staticmethod
    def parse_pdf(
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ) -> List[TextSegment]:
        """
        Loads a PDF and extracts structured TextSegment objects using
        RecursiveCharacterTextSplitter for paragraph-level division.
        """
        if not doc_id:
            doc_id = os.path.basename(file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        segments: List[TextSegment] = []
        seq_idx = 0

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for idx, page in enumerate(pages):
            text_content = page.page_content or ""
            if not text_content.strip():
                continue

            page_num = page.metadata.get("page", idx) + 1
            chunks = splitter.split_text(text_content)

            for chunk in chunks:
                if chunk.strip():
                    segments.append(
                        TextSegment(
                            content=chunk.strip(),
                            doc_id=doc_id,
                            page_num=page_num,
                            seq_index=seq_idx
                        )
                    )
                    seq_idx += 1

        return segments

class EmbeddingGenerator:
    """Thread-safe lazy loading container for SentenceTransformers models."""
    
    @staticmethod
    def get_model(model_name: str):
        global _model_registry
        if model_name not in _model_registry:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            _model_registry[model_name] = SentenceTransformer(model_name)
        return _model_registry[model_name]

    @classmethod
    def embed_texts(cls, texts: List[str], model_name: str = DEFAULT_ENCODER) -> List[List[float]]:
        """Converts a collection of strings into vector arrays."""
        if not texts:
            return []
        
        transformer = cls.get_model(model_name)
        embeddings = transformer.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
