# config.py
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

# ─── Selector de modo ──────────────────────────────────────────────────────────
# True  = Modo B: MarkItDown + LangExtract (más preciso, más lento, más costoso)
# False = Modo A: solo MarkItDown           (recomendado para empezar)
USE_LANGEXTRACT = True

# ─── Rutas ────────────────────────────────────────────────────────────────────
CHUNKS_SIMPLE_PATH   = _os.path.join(_ROOT, "data/processed/chunks_simple/all_chunks.json")
CHUNKS_ENRICHED_PATH = _os.path.join(_ROOT, "data/processed/chunks_enriched/all_chunks.json")

CHROMA_PATH_SIMPLE   = _os.path.join(_ROOT, "chroma_db/memorias_simple")
CHROMA_PATH_ENRICHED = _os.path.join(_ROOT, "chroma_db/memorias_enriched")

COLLECTION_SIMPLE    = "memorias_simple"
COLLECTION_ENRICHED  = "memorias_enriched"

# ─── Modelos ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "gemini-embedding-001"
EMBEDDING_DIM     = 768
LLM_MODEL         = "gemini-2.5-flash"
LANGEXTRACT_MODEL = "gemini-2.5-flash"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 300

# ─── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_K        = 15   # candidatos iniciales del vector store
RERANK_TOP_N       = 7    # chunks finales tras reranking
USE_RERANKING      = True  # True = rerank por lotes con LLM; False = solo embedding similarity
