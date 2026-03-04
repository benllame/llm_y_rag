"""
Extracción estructurada con LangExtract + Gemini (OPCIONAL).
Activa con USE_LANGEXTRACT = True en config.py.
El chunker.py detecta automáticamente el modo y usa este output si está disponible.
"""
import langextract as lx
import json
import os
import textwrap
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Agregar el directorio raíz al path para poder importar config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import LANGEXTRACT_MODEL

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = textwrap.dedent("""\
    Extrae indicadores cuantitativos y cualitativos de esta Memoria Anual
    Integrada bajo la NCG 461 de la CMF (Chile).

    Clases de entidades a extraer:
    - indicador_financiero: ingresos, EBITDA, utilidad neta, deuda/equity, ROE
    - indicador_ambiental: emisiones GEI scope 1/2/3, consumo de agua y energía, residuos
    - indicador_social: número de empleados, brecha salarial, accidentabilidad, capacitación
    - indicador_gobernanza: composición del directorio, % mujeres directivas, remuneraciones
    - riesgo_identificado: riesgos materiales (climáticos, regulatorios, operacionales)

    Usa el texto EXACTO del documento como extraction_text.
    Incluye el valor numérico, unidad y año en los atributos.
""")

EXTRACTION_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "| Indicador                  | 2022    | 2023    |\n"
            "| Emisiones Scope 1 (tCO2eq) | 50.000  | 45.230  |\n"
            "| Brecha salarial (%)        | 20.1    | 18.3    |"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="indicador_ambiental",
                extraction_text="| Emisiones Scope 1 (tCO2eq) | 50.000  | 45.230  |",
                attributes={"nombre": "emisiones_scope1", "valor": "45230", "unidad": "tCO2eq", "año": "2023"},
            ),
            lx.data.Extraction(
                extraction_class="indicador_social",
                extraction_text="| Brecha salarial (%)        | 20.1    | 18.3    |",
                attributes={"nombre": "brecha_salarial", "valor": "18.3", "unidad": "%", "año": "2023"},
            ),
        ],
    ),
]


def extract_structured_entities(document: dict) -> dict:
    logger.info(f"Extrayendo entidades de {document['source_id']}...")
    result = lx.extract(
        text_or_documents=document["full_text_md"],
        prompt_description=EXTRACTION_PROMPT,
        examples=EXTRACTION_EXAMPLES,
        model_id=LANGEXTRACT_MODEL,
        extraction_passes=2,
        max_workers=10,
    )
    entities = [
        {
            "entity_type": e.extraction_class,
            "text":        e.extraction_text,
            "attributes":  e.attributes or {},
            "company":     document["company"],
            "year":        document["year"],
            "source_id":   document["source_id"],
        }
        for e in result.extractions
    ]
    logger.info(f"  ✅ {len(entities)} entidades extraídas de {document['source_id']}")
    return {"source_id": document["source_id"], "company": document["company"],
            "year": document["year"], "entities": entities}


def extract_all_structured(
    input_folder: str = "data/processed/markdown",
    output_folder: str = "data/processed/langextract_output",
):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for json_file in sorted(Path(input_folder).glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            document = json.load(f)
        out_path = Path(output_folder) / f"{document['source_id']}_structured.json"
        if out_path.exists():
            logger.info(f"Saltando {document['source_id']} (ya procesado)")
            continue
        result = extract_structured_entities(document)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("✅ Extracción estructurada completada")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Configura GOOGLE_API_KEY en .env")
        exit(1)
    extract_all_structured()
