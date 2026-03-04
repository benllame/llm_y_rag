"""
Chunking semántico con soporte para ambos modos.

Modo A (USE_LANGEXTRACT=False):
  → Solo chunks narrativos del Markdown
  → Output: data/processed/chunks_simple/all_chunks.json

Modo B (USE_LANGEXTRACT=True):
  → Chunks narrativos + chunks de entidades LangExtract
  → Output: data/processed/chunks_enriched/all_chunks.json
"""
import json
import re
import logging
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para poder importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    USE_LANGEXTRACT, CHUNK_SIZE, CHUNK_OVERLAP,
    CHUNKS_SIMPLE_PATH, CHUNKS_ENRICHED_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Separadores Markdown-aware (de más específico a más genérico) ────────────
_MD_SEPARATORS = [
    "\n## ",     # Sección H2
    "\n### ",    # Subsección H3
    "\n#### ",   # H4
    "\n\n",      # Doble salto (párrafo)
    "\n",        # Salto simple
    ". ",        # Oración
    " ",         # Espacio
]

# ─── Regex para detectar encabezados Markdown activos ─────────────────────────
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)


def _extract_active_heading(text: str) -> str:
    """Extrae el último heading visible antes del final del texto."""
    matches = list(_HEADING_RE.finditer(text))
    return matches[-1].group(0).strip() if matches else ""


def _get_section_context(full_page_text: str, chunk_start: str) -> str:
    """Busca el heading de sección más reciente antes de donde empieza el chunk."""
    pos = full_page_text.find(chunk_start[:80])
    if pos <= 0:
        return ""
    preceding = full_page_text[:pos]
    matches = list(_HEADING_RE.finditer(preceding))
    return matches[-1].group(0).strip() if matches else ""


def chunk_markdown_document(document: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_MD_SEPARATORS,
        keep_separator=True,
    )

    company = document["company"]
    year    = document["year"]
    doc_header = f"[{company} — Memoria Anual {year}]"

    # Procesar página por página si la estructura está disponible
    if "pages" in document:
        chunks = []
        for p in document["pages"]:
            page_num  = p["page_num"]
            page_text = p["text_md"].strip()
            if not page_text:
                continue

            for i, chunk_text in enumerate(splitter.split_text(page_text)):
                if len(chunk_text.strip()) < 50:
                    continue
                # Inyectar contexto: heading de sección + header del documento
                section = _get_section_context(page_text, chunk_text)
                prefix_parts = [doc_header]
                if section:
                    prefix_parts.append(section)
                prefix = "\n".join(prefix_parts) + "\n\n"

                # Solo agregar prefijo si el chunk no ya lo contiene
                if not chunk_text.strip().startswith(("#", doc_header)):
                    enriched_text = prefix + chunk_text
                else:
                    enriched_text = chunk_text

                chunks.append({
                    "chunk_id":   f"{document['source_id']}_p{page_num:04d}_{i:03d}",
                    "text":       enriched_text,
                    "chunk_type": "narrative_md",
                    "company":    company,
                    "year":       year,
                    "source_id":  document["source_id"],
                    "page_num":   page_num,
                    "section":    section,
                })
        return chunks
    else:
        # Fallback por si hay extracciones antiguas sin páginas
        text = document.get("full_text_md", "").strip()
        if not text:
            return []
        return [
            {
                "chunk_id":   f"{document['source_id']}_md_{i:04d}",
                "text":       f"{doc_header}\n\n{chunk_text}",
                "chunk_type": "narrative_md",
                "company":    company,
                "year":       year,
                "source_id":  document["source_id"],
            }
            for i, chunk_text in enumerate(splitter.split_text(text))
            if len(chunk_text.strip()) >= 50
        ]


def langextract_entities_to_chunks(structured_result: dict) -> list[dict]:
    chunks = []
    for i, entity in enumerate(structured_result.get("entities", [])):
        attrs_text = ""
        if entity.get("attributes"):
            lines = [f"  {k}: {v}" for k, v in entity["attributes"].items()]
            attrs_text = "\nAtributos:\n" + "\n".join(lines)
        chunk_text = (
            f"[{entity['entity_type'].upper()}]\n"
            f"Empresa: {entity['company']} | Año: {entity['year']}\n"
            f"Texto fuente: {entity['text']}{attrs_text}"
        )
        chunks.append({
            "chunk_id":    f"{entity['source_id']}_entity_{i:04d}",
            "text":        chunk_text,
            "chunk_type":  "structured_entity",
            "entity_type": entity["entity_type"],
            "company":     entity["company"],
            "year":        entity["year"],
            "source_id":   entity["source_id"],
        })
    return chunks


def build_all_chunks(
    md_folder: str = "data/processed/markdown",
    structured_folder: str = "data/processed/langextract_output",
) -> str:
    """Construye el JSON de chunks según el modo activo. Retorna la ruta generada."""
    output_path = CHUNKS_ENRICHED_PATH if USE_LANGEXTRACT else CHUNKS_SIMPLE_PATH
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_chunks, narrative_count, entity_count = [], 0, 0

    for json_file in sorted(Path(md_folder).glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            document = json.load(f)
        new = chunk_markdown_document(document)
        all_chunks.extend(new)
        narrative_count += len(new)
        logger.info(f"  {document['source_id']}: {len(new)} chunks narrativos")

    if USE_LANGEXTRACT:
        structured_path = Path(structured_folder)
        if not structured_path.exists() or not list(structured_path.glob("*_structured.json")):
            logger.warning(
                "USE_LANGEXTRACT=True pero no hay archivos en langextract_output/. "
                "Ejecuta primero: python src/extract_structured.py"
            )
        else:
            for json_file in sorted(structured_path.glob("*_structured.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    structured_result = json.load(f)
                new = langextract_entities_to_chunks(structured_result)
                all_chunks.extend(new)
                entity_count += len(new)
                logger.info(f"  {structured_result['source_id']}: {len(new)} chunks de entidades")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    modo = "ENRIQUECIDO (Modo B)" if USE_LANGEXTRACT else "SIMPLE (Modo A)"
    logger.info(f"\n✅ {modo}: {len(all_chunks)} chunks ({narrative_count} narrativos + {entity_count} entidades)")
    return output_path


if __name__ == "__main__":
    build_all_chunks()
