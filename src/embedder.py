"""
Indexado en ChromaDB con el SDK google-genai.

CORRECCIONES APLICADAS:
1. __init__ explícito en GeminiEmbeddingFunction (evita DeprecationWarning).
2. task_type="RETRIEVAL_DOCUMENT" forzado para indexado.
3. La query usa RETRIEVAL_QUERY en chain.py, no aquí.
4. output_dimensionality=768 activa MRL (reduce espacio sin perder calidad).
"""
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
import json
import os
import sys
import time
import logging
from dotenv import load_dotenv

# Agregar el directorio raíz al path para poder importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    USE_LANGEXTRACT,
    CHUNKS_SIMPLE_PATH, CHUNKS_ENRICHED_PATH,
    CHROMA_PATH_SIMPLE, CHROMA_PATH_ENRICHED,
    COLLECTION_SIMPLE, COLLECTION_ENRICHED,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = EMBEDDING_MODEL
        self.task_type = "RETRIEVAL_DOCUMENT"
        self.output_dimensionality = EMBEDDING_DIM

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=input,
                config=types.EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=self.output_dimensionality,
                ),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
            return [[0.0] * self.output_dimensionality] * len(input)


def build_vectorstore(mode: str = "auto", batch_size: int = 50) -> chromadb.Collection:
    """
    Construye el vector store para el modo indicado.
    mode: "simple" | "enriched" | "auto"
    """
    if mode == "auto":
        mode = "enriched" if USE_LANGEXTRACT else "simple"

    chunks_path = CHUNKS_ENRICHED_PATH if mode == "enriched" else CHUNKS_SIMPLE_PATH
    chroma_path = CHROMA_PATH_ENRICHED  if mode == "enriched" else CHROMA_PATH_SIMPLE
    coll_name   = COLLECTION_ENRICHED   if mode == "enriched" else COLLECTION_SIMPLE

    logger.info(f"Indexando en modo: {mode.upper()} → '{coll_name}'")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=coll_name,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "company":    c["company"],
                    "year":       c["year"],
                    "chunk_type": c.get("chunk_type", "narrative_md"),
                    "source_id":  c.get("source_id", ""),
                    "page_num":   c.get("page_num", -1),
                }
                for c in batch
            ],
        )
        time.sleep(0.1)
        logger.info(f"  [{min(i+batch_size,total)/total*100:.0f}%] {min(i+batch_size,total)}/{total}")

    logger.info(f"✅ {collection.count()} chunks indexados en '{chroma_path}'")
    return collection


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY no encontrada")
        exit(1)
    build_vectorstore(mode="auto")
