"""
Extracción de PDFs a Markdown usando MarkItDown (Microsoft).

Para mayor precisión y mejor metadata (número de página), el PDF se
procesa hoja por hoja separándolo mediante pypdf.
"""
import json
import logging
import tempfile
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from markitdown import MarkItDown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Instancia global reutilizable (evita re-inicializar en cada llamada)
_md_instance = None


def get_markitdown() -> MarkItDown:
    """Inicialización lazy del extractor (singleton)."""
    global _md_instance
    if _md_instance is None:
        _md_instance = MarkItDown()
    return _md_instance


COMPANY_MAP = {
    "cf": "CF Seguros de Vida",
}


def parse_filename(filename: str) -> dict:
    """Extrae company y year del nombre del archivo (convención Empresa_Año.pdf)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    raw_company = parts[0] if parts else stem
    return {
        "company": COMPANY_MAP.get(raw_company.lower(), raw_company),
        "year": parts[1] if len(parts) > 1 else "unknown",
    }


def extract_pdf_to_markdown(pdf_path: str) -> dict | None:
    """
    Convierte un PDF a Markdown iterando página por página.
    
    Retorna un dict con:
      - source_id:    str  (clave única "Empresa_Año")
      - company:      str
      - year:         str
      - total_pages:  int
      - total_chars:  int
      - full_text_md: str
      - pages:        list[dict]
    """
    pdf_path = Path(pdf_path)
    meta = parse_filename(pdf_path.name)

    logger.info(f"Convirtiendo {pdf_path.name} hoja por hoja con MarkItDown...")

    try:
        md = get_markitdown()
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        
        pages_data = []
        full_text_parts = []
        
        for i in range(num_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                
            try:
                writer.write(tmp_path)
                result = md.convert(tmp_path)
                page_text = result.text_content.strip() if result.text_content else ""
                
                if page_text:
                    pages_data.append({"page_num": i + 1, "text_md": page_text})
                    full_text_parts.append(f"## --- PÁGINA {i + 1} ---\n\n{page_text}")
            except Exception as e:
                logger.error(f"Error en página {i+1} de {pdf_path.name}: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
            if (i + 1) % 10 == 0 or (i + 1) == num_pages:
                logger.info(f"  Procesadas {i + 1}/{num_pages} páginas de {pdf_path.name}")
                
        full_text_md = "\n\n".join(full_text_parts)
        
    except Exception as e:
        logger.error(f"Error procesando {pdf_path.name}: {e}")
        return None

    if not full_text_md or len(full_text_md) < 200:
        logger.warning(
            f"Texto extraído muy corto en {pdf_path.name} "
            f"({len(full_text_md)} chars). ¿Es un PDF escaneado?"
        )
        return None

    logger.info(f"  ✅ {len(full_text_md):,} caracteres extraídos de {pdf_path.name}")

    return {
        "filename":     pdf_path.name,
        "source_id":    f"{meta['company']}_{meta['year']}",
        "company":      meta["company"],
        "year":         meta["year"],
        "total_pages":  num_pages,
        "total_chars":  len(full_text_md),
        "full_text_md": full_text_md,
        "pages":        pages_data
    }


def process_all_pdfs(
    pdf_folder: str = "data/raw/pdfs",
    output_folder: str = "data/processed/markdown",
):
    """Procesa todos los PDFs de la carpeta y guarda el Markdown en JSON."""
    pdf_folder = Path(pdf_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No se encontraron PDFs en {pdf_folder}")
        return

    logger.info(f"Procesando {len(pdf_files)} PDFs con MarkItDown...")
    processed = 0

    for pdf_file in pdf_files:
        out_path = output_folder / f"{pdf_file.stem}.json"
        if out_path.exists():
            logger.info(f"Saltando {pdf_file.name} (ya procesado)")
            continue

        document = extract_pdf_to_markdown(str(pdf_file))
        if document:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            processed += 1

    logger.info(f"✅ Extracción completada: {processed} archivos nuevos procesados")


if __name__ == "__main__":
    process_all_pdfs()
