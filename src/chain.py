"""
RAG Chain: Recuperación → Augmentación → Generación.
"""
import chromadb
from google import genai
from google.genai import types
import os
import sys
import logging
from dotenv import load_dotenv

# Agregar el directorio raíz al path para poder importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    USE_LANGEXTRACT, USE_RERANKING,
    CHROMA_PATH_SIMPLE, CHROMA_PATH_ENRICHED,
    COLLECTION_SIMPLE, COLLECTION_ENRICHED,
    EMBEDDING_MODEL, EMBEDDING_DIM, LLM_MODEL,
    RETRIEVAL_K, RERANK_TOP_N,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analista financiero especializado en Memorias Anuales Integradas
bajo la Norma de Carácter General N°461 de la CMF (Chile).

Responde ÚNICAMENTE con información de las fuentes provistas.

REGLAS:
1. Cita siempre la empresa, el año, el tipo de fuente y la PÁGINA (si está disponible)
2. Si el dato está en una tabla Markdown, analiza sus columnas con cuidado
3. Si un dato no está en el contexto: "No tengo esa información disponible"
4. NUNCA inventes números, fechas o hechos
5. Para comparaciones, usa tablas o listas estructuradas
6. Incluye siempre la unidad de los indicadores (tCO2eq, %, millones de pesos, UF)
7. Responde en español
8. Incluye siempre los VALORES NUMÉRICOS EXACTOS que aparecen en las fuentes
9. Cuando la pregunta pida montos, cita el número completo con su unidad (M$, UF, %, etc.)"""

BATCH_RERANK_PROMPT = """Eres un evaluador de relevancia para un sistema RAG.
Dada una PREGUNTA y una lista de FRAGMENTOS numerados, asigna un puntaje de 0 a 10
a cada fragmento según qué tan útil es para responder la pregunta.

- 0: Completamente irrelevante
- 3: Menciona el tema pero no responde
- 5: Parcialmente relevante
- 8: Muy relevante, contiene datos clave
- 10: Contiene la respuesta exacta

Responde SOLO con el formato (una línea por fragmento, sin texto adicional):
1: <puntaje>
2: <puntaje>
...

PREGUNTA: {question}

{fragments}"""


class MemoriasRAG:
    def __init__(self, mode: str = "auto"):
        if mode == "auto":
            mode = "enriched" if USE_LANGEXTRACT else "simple"
        self.mode = mode
        chroma_path = CHROMA_PATH_ENRICHED if mode == "enriched" else CHROMA_PATH_SIMPLE
        coll_name   = COLLECTION_ENRICHED  if mode == "enriched" else COLLECTION_SIMPLE

        self.gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.llm_model = LLM_MODEL
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        # SIN embedding_function → control manual del task_type
        self.collection = chroma_client.get_collection(coll_name)
        logger.info(f"RAG | Modo: {mode.upper()} | {self.collection.count()} chunks")

    def _embed_query(self, query: str) -> list[float]:
        response = self.gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return response.embeddings[0].values

    def retrieve(self, query: str, n_results: int = None, filters: dict = None) -> list[dict]:
        """
        Filtros ChromaDB:
          {"company": "Falabella"}
          {"$and": [{"company": "BCI"}, {"year": "2023"}]}
        """
        if n_results is None:
            n_results = RETRIEVAL_K
        query_vector = self._embed_query(query)
        params = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            params["where"] = filters
        try:
            results = self.collection.query(**params)
        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return []
        chunks = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                chunks.append({"text": doc, "metadata": meta, "similarity": round(1-dist, 4)})
        return sorted(chunks, key=lambda x: x["similarity"], reverse=True)

    def rerank(self, question: str, chunks: list[dict], top_n: int = None) -> list[dict]:
        """Re-puntúa chunks en UNA sola llamada LLM (batch reranking)."""
        if top_n is None:
            top_n = RERANK_TOP_N
        if len(chunks) <= top_n:
            return chunks

        # Construir prompt con todos los fragmentos numerados
        fragment_parts = []
        for idx, c in enumerate(chunks, 1):
            # Limitar cada chunk a ~800 chars para que quepan todos en un solo prompt
            text_preview = c["text"][:800].replace("\n", " ").strip()
            fragment_parts.append(f"--- FRAGMENTO {idx} ---\n{text_preview}")
        fragments_block = "\n\n".join(fragment_parts)

        prompt = BATCH_RERANK_PROMPT.format(
            question=question,
            fragments=fragments_block,
        )

        try:
            resp = self.gemini.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                ),
            )
            # Parsear respuesta: "1: 8\n2: 3\n..."
            import re
            scores_map = {}
            for line in resp.text.strip().splitlines():
                m = re.match(r"(\d+)\s*:\s*([\d.]+)", line.strip())
                if m:
                    frag_idx = int(m.group(1))
                    score = min(10.0, max(0.0, float(m.group(2))))
                    scores_map[frag_idx] = score

            scored = []
            for idx, c in enumerate(chunks, 1):
                score = scores_map.get(idx, c["similarity"] * 10)
                scored.append({**c, "rerank_score": score})

        except Exception as e:
            logger.warning(f"Batch rerank error: {e} — usando similarity como fallback")
            scored = [{**c, "rerank_score": c["similarity"] * 10} for c in chunks]

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            m = c["metadata"]
            tipo = "ESTRUCTURADO" if m.get("chunk_type") == "structured_entity" else "MARKDOWN"
            pagina_val = m.get("page_num", -1)
            pagina = f" | Pág: {pagina_val}" if pagina_val != -1 else ""
            header = f"[FUENTE {i} | {tipo} | {m.get('company','?')} {m.get('year','?')}{pagina} | Sim: {c['similarity']}]"
            parts.append(f"{header}\n{c['text']}")
        return "\n\n" + ("─"*60 + "\n\n").join(parts)

    def answer(self, question: str, n_results: int = None, filters: dict = None,
               temperature: float = 0.1, use_rerank: bool = None) -> dict:
        if use_rerank is None:
            use_rerank = USE_RERANKING

        candidates = self.retrieve(question, n_results=n_results, filters=filters)
        if not candidates:
            return {"answer": "No encontré información relevante.", "sources": [],
                    "model": self.llm_model, "mode": self.mode, "n_chunks": 0}

        if use_rerank:
            chunks = self.rerank(question, candidates)
        else:
            chunks = candidates[:RERANK_TOP_N]

        context = self._build_context(chunks)
        user_prompt = f"CONTEXTO:\n{context}\n\n{'─'*60}\n\nPREGUNTA: {question}\n\nResponde solo con el contexto anterior."

        response = self.gemini.models.generate_content(
            model=self.llm_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                max_output_tokens=2000,
            ),
        )
        return {"answer": response.text, "sources": chunks, "model": self.llm_model, "mode": self.mode, "n_chunks": len(chunks)}

    def answer_stream(
        self,
        question: str,
        n_results: int = None,
        filters: dict = None,
        temperature: float = 0.1,
        use_rerank: bool = None,
    ) -> tuple:
        """
        Streaming version of answer().

        Retrieval + reranking run synchronously before the generator is returned,
        so the caller can show a spinner during that phase and then hand the
        generator to st.write_stream() for live token rendering.

        Returns
        -------
        (text_generator, metadata_dict)
            text_generator  : Generator[str, None, None] — yields tokens as they arrive.
            metadata_dict   : {"sources", "mode", "n_chunks", "model"}
        """
        if use_rerank is None:
            use_rerank = USE_RERANKING

        # ── Retrieval + Reranking (blocking) ─────────────────────────────────
        candidates = self.retrieve(question, n_results=n_results, filters=filters)
        if not candidates:
            def _empty():
                yield "No encontré información relevante para esta pregunta."
            return _empty(), {"sources": [], "mode": self.mode, "n_chunks": 0, "model": self.llm_model}

        chunks = self.rerank(question, candidates) if use_rerank else candidates[:RERANK_TOP_N]

        # ── Build prompt ──────────────────────────────────────────────────────
        context = self._build_context(chunks)
        user_prompt = (
            f"CONTEXTO:\n{context}\n\n{'─'*60}\n\n"
            f"PREGUNTA: {question}\n\nResponde solo con el contexto anterior."
        )

        # ── Lazy streaming generator ──────────────────────────────────────────
        # NOTE: the HTTP call to Gemini only starts when this generator is iterated.
        def _stream_tokens():
            try:
                for chunk in self.gemini.models.generate_content_stream(
                    model=self.llm_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                        max_output_tokens=2000,
                    ),
                ):
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"\n\n⚠️ Error durante la generación: {e}"

        metadata = {
            "sources": chunks,
            "mode": self.mode,
            "n_chunks": len(chunks),
            "model": self.llm_model,
        }
        return _stream_tokens(), metadata
