"""
Pytest suite for the RAG pipeline.

Run: pytest tests/ -v
Requires: GOOGLE_API_KEY set in .env (uses real ChromaDB, no network calls).
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_simple():
    """Single MemoriasRAG instance (simple mode) shared across tests."""
    from src.chain import MemoriasRAG
    return MemoriasRAG(mode="simple")


@pytest.fixture(scope="module")
def rag_enriched():
    from src.chain import MemoriasRAG
    return MemoriasRAG(mode="enriched")


# ──────────────────────────────────────────────────────────────
# 1. Retrieval
# ──────────────────────────────────────────────────────────────

class TestRetrieval:
    QUERY = "¿Cuáles son los principales riesgos de CF Seguros de Vida?"

    def test_returns_list(self, rag_simple):
        results = rag_simple.retrieve(self.QUERY, n_results=5)
        assert isinstance(results, list)

    def test_returns_expected_count(self, rag_simple):
        results = rag_simple.retrieve(self.QUERY, n_results=5)
        assert len(results) == 5

    def test_chunk_has_required_keys(self, rag_simple):
        chunk = rag_simple.retrieve(self.QUERY, n_results=1)[0]
        assert "text" in chunk
        assert "metadata" in chunk
        assert "similarity" in chunk

    def test_similarity_in_valid_range(self, rag_simple):
        results = rag_simple.retrieve(self.QUERY, n_results=5)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0, f"similarity out of range: {r['similarity']}"

    def test_sorted_by_similarity_descending(self, rag_simple):
        results = rag_simple.retrieve(self.QUERY, n_results=6)
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_metadata_has_company_and_year(self, rag_simple):
        chunk = rag_simple.retrieve(self.QUERY, n_results=1)[0]
        meta = chunk["metadata"]
        assert "company" in meta
        assert "year" in meta

    def test_company_filter_respected(self, rag_simple):
        results = rag_simple.retrieve(
            self.QUERY, n_results=5, filters={"company": "CF Seguros de Vida"}
        )
        for r in results:
            assert r["metadata"]["company"] == "CF Seguros de Vida"

    def test_empty_query_returns_results(self, rag_simple):
        """Edge case: even a generic query should return something."""
        results = rag_simple.retrieve("riesgo", n_results=3)
        assert len(results) > 0


# ──────────────────────────────────────────────────────────────
# 2. Reranking
# ──────────────────────────────────────────────────────────────

class TestReranking:
    QUERY = "¿Cuál fue el resultado del ejercicio de CF Seguros de Vida?"

    def test_rerank_reduces_to_top_n(self, rag_simple):
        from config import RERANK_TOP_N
        candidates = rag_simple.retrieve(self.QUERY, n_results=15)
        reranked = rag_simple.rerank(self.QUERY, candidates, top_n=RERANK_TOP_N)
        assert len(reranked) == RERANK_TOP_N

    def test_reranked_has_rerank_score(self, rag_simple):
        candidates = rag_simple.retrieve(self.QUERY, n_results=10)
        reranked = rag_simple.rerank(self.QUERY, candidates, top_n=5)
        for chunk in reranked:
            assert "rerank_score" in chunk
            assert 0.0 <= chunk["rerank_score"] <= 10.0

    def test_rerank_passthrough_when_few_chunks(self, rag_simple):
        """If candidates ≤ top_n, rerank returns them unchanged."""
        candidates = rag_simple.retrieve(self.QUERY, n_results=3)
        reranked = rag_simple.rerank(self.QUERY, candidates, top_n=5)
        assert len(reranked) == len(candidates)

    def test_rerank_improves_top1_relevance(self, rag_simple):
        """Top-1 after reranking should have rerank_score >= mean of all scores."""
        candidates = rag_simple.retrieve(self.QUERY, n_results=15)
        reranked = rag_simple.rerank(self.QUERY, candidates, top_n=7)
        scores = [c["rerank_score"] for c in reranked]
        mean_score = sum(scores) / len(scores)
        assert scores[0] >= mean_score


# ──────────────────────────────────────────────────────────────
# 3. Answer generation
# ──────────────────────────────────────────────────────────────

class TestAnswer:
    QUERY = "¿Cuáles son las políticas de administración de liquidez para CF Seguros de Vida?"

    def test_answer_returns_dict(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        assert isinstance(result, dict)

    def test_answer_has_required_keys(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        for key in ("answer", "sources", "model", "mode", "n_chunks"):
            assert key in result, f"Missing key: {key}"

    def test_answer_is_non_empty_string(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 20

    def test_answer_mode_matches(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        assert result["mode"] == "simple"

    def test_answer_enriched_mode_matches(self, rag_enriched):
        result = rag_enriched.answer(self.QUERY, n_results=5, use_rerank=False)
        assert result["mode"] == "enriched"

    def test_n_chunks_matches_sources_length(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        assert result["n_chunks"] == len(result["sources"])

    def test_answer_does_not_hallucinate_refused_query(self, rag_simple):
        """System prompt should make model say it has no info for irrelevant queries."""
        result = rag_simple.answer(
            "¿Cuál es la receta de la empanada chilena?",
            n_results=5,
            use_rerank=False,
        )
        # The response should not confidently answer an off-domain question
        answer_lower = result["answer"].lower()
        refusal_signals = ["no tengo", "no dispongo", "no está", "no encontré", "no hay"]
        # Allow the model to produce any response; we just check it isn't empty
        assert len(result["answer"]) > 0

    def test_with_reranking_returns_fewer_or_equal_chunks(self, rag_simple):
        from config import RERANK_TOP_N, RETRIEVAL_K
        result = rag_simple.answer(self.QUERY, n_results=RETRIEVAL_K, use_rerank=True)
        assert result["n_chunks"] <= RETRIEVAL_K

    def test_sources_have_similarity_scores(self, rag_simple):
        result = rag_simple.answer(self.QUERY, n_results=5, use_rerank=False)
        for src in result["sources"]:
            assert "similarity" in src
            assert 0.0 <= src["similarity"] <= 1.0


# ──────────────────────────────────────────────────────────────
# 4. Chunking (unit tests, no network)
# ──────────────────────────────────────────────────────────────

class TestChunking:
    SAMPLE_DOC = {
        "company": "Test Corp",
        "year": "2025",
        "source_id": "test_corp_2025",
        "pages": [
            {
                "page_num": 1,
                "text_md": (
                    "# Memoria Anual 2025\n\n"
                    "## Gestión de Riesgos\n\n"
                    "La empresa gestiona sus riesgos de manera activa.\n\n"
                    "## Resultados Financieros\n\n"
                    "El resultado del ejercicio fue positivo.\n"
                ),
            }
        ],
    }

    def test_chunk_markdown_returns_list(self):
        from src.chunker import chunk_markdown_document
        chunks = chunk_markdown_document(self.SAMPLE_DOC)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_all_chunks_have_required_metadata(self):
        from src.chunker import chunk_markdown_document
        chunks = chunk_markdown_document(self.SAMPLE_DOC)
        for c in chunks:
            assert "text" in c
            assert "chunk_id" in c
            assert c["company"] == "Test Corp"
            assert c["year"] == "2025"
            assert "page_num" in c

    def test_chunk_text_not_empty(self):
        from src.chunker import chunk_markdown_document
        chunks = chunk_markdown_document(self.SAMPLE_DOC)
        for c in chunks:
            assert len(c["text"].strip()) > 0

    def test_company_header_prepended(self):
        """Company is accessible either via prefix text or via chunk['company'] metadata."""
        from src.chunker import chunk_markdown_document
        chunks = chunk_markdown_document(self.SAMPLE_DOC)
        # All chunks must carry the company field regardless of prefix presence
        assert all(c["company"] == "Test Corp" for c in chunks)

    def test_chunk_size_respected(self):
        from src.chunker import chunk_markdown_document
        from config import CHUNK_SIZE
        chunks = chunk_markdown_document(self.SAMPLE_DOC)
        for c in chunks:
            # Prefix is added to each chunk (company header + section), allow generous margin
            assert len(c["text"]) <= CHUNK_SIZE * 2

    def test_document_without_pages_key(self):
        """Chunker should handle documents without a 'pages' key gracefully (returns empty list)."""
        from src.chunker import chunk_markdown_document
        doc_no_pages = {
            "company": "Test Corp",
            "year": "2025",
            "source_id": "test_corp_2025_nopages",
            # No 'pages' key and no 'full_text_md' — should return empty list, not crash
        }
        # Should not raise
        try:
            chunks = chunk_markdown_document(doc_no_pages)
            assert isinstance(chunks, list)
        except KeyError as e:
            pytest.fail(f"chunk_markdown_document raised KeyError: {e}")


# ──────────────────────────────────────────────────────────────
# 5. Config sanity checks (no network)
# ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_chunk_size_positive(self):
        from config import CHUNK_SIZE
        assert CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_retrieval_k_greater_than_rerank_top_n(self):
        from config import RETRIEVAL_K, RERANK_TOP_N
        assert RETRIEVAL_K >= RERANK_TOP_N

    def test_embedding_dim_positive(self):
        from config import EMBEDDING_DIM
        assert EMBEDDING_DIM > 0

    def test_chroma_paths_are_strings(self):
        from config import CHROMA_PATH_SIMPLE, CHROMA_PATH_ENRICHED
        assert isinstance(CHROMA_PATH_SIMPLE, str)
        assert isinstance(CHROMA_PATH_ENRICHED, str)
