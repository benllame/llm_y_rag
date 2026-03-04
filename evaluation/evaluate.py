"""
Evaluación comparativa RAGAS: Modo A (simple) vs Modo B (enriched).

El script corre las mismas preguntas contra ambas colecciones ChromaDB
y genera una tabla comparativa con delta entre modos.

Prerequisito: tener ambas colecciones indexadas.
  Con USE_LANGEXTRACT=False: python src/embedder.py  (genera memorias_simple)
  Con USE_LANGEXTRACT=True:  python src/embedder.py  (genera memorias_enriched)
"""
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings import GoogleEmbeddings
import instructor
from google import genai
from src.chain import MemoriasRAG

# ── Gemini LLM + Embeddings para RAGAS (usa google-genai SDK nativo) ─────────
_google_api_key = os.getenv("GOOGLE_API_KEY")
if not _google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY no está definida en .env")

# LLM: google-genai nativo con instructor async (sin OpenAI)
_genai_client = genai.Client(api_key=_google_api_key)
_async_instructor = instructor.from_genai(_genai_client, use_async=True)
_ragas_llm = InstructorLLM(
    client=_async_instructor,
    model="gemini-2.5-flash",
    provider="google",
    model_args=InstructorModelArgs(),
)

# Embeddings: genai.Client sync; GoogleEmbeddings lo envuelve en thread pool para async.
_ragas_embeddings = GoogleEmbeddings(client=_genai_client, model="gemini-embedding-001")

# ── Ground Truths — extraídos MANUALMENTE de los PDFs ─────────────────────────



TEST_QUESTIONS = [
    # ── Financiero — numérico ─────────────────────────────────────────────────
    {
        "question": "¿Cuál fue el resultado del ejercicio de CF Seguros de Vida al 31 de diciembre de 2025?",
        "ground_truth": (
            "El resultado del ejercicio al 31 de diciembre de 2025 fue de M$ 18.299.051 "
            "(aproximadamente 18.299 millones de pesos chilenos), comparado con M$ 13.255.625 "
            "en 2024, lo que representa un aumento significativo año a año."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿A cuánto ascendió el total de patrimonio de CF Seguros de Vida al cierre de 2025?",
        "ground_truth": (
            "El total de patrimonio al 31 de diciembre de 2025 fue de M$ 20.647.153, "
            "compuesto por capital pagado de M$ 2.483.231, resultados acumulados de M$ 18.027.208 "
            "(que incluyen resultado del ejercicio de M$ 18.299.051 y dividendos pagados de "
            "M$ -12.856.960), y otros ajustes de M$ 136.714."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuánto totalizaron las reservas técnicas de CF Seguros de Vida en 2025?",
        "ground_truth": (
            "Las reservas técnicas totalizaron M$ 25.937.184 al 31 de diciembre de 2025, "
            "compuestas principalmente por: reserva de riesgos en curso M$ 7.482.162, "
            "reserva matemática M$ 8.647.754, reserva de siniestros M$ 9.642.817, "
            "y reserva de insuficiencia de prima M$ 164.451."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuál fue el total de pasivos de CF Seguros de Vida al 31 de diciembre de 2025?",
        "ground_truth": (
            "El total de pasivos al 31 de diciembre de 2025 fue de M$ 29.212.811, frente a "
            "M$ 27.486.404 en 2024. Los principales componentes son: cuentas pasivos de seguros "
            "por M$ 26.455.372 (que incluyen reservas técnicas y deudas por operaciones de seguro) "
            "y otros pasivos por M$ 2.757.439."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuál fue el total del activo de CF Seguros de Vida al cierre del ejercicio 2025?",
        "ground_truth": (
            "El total de pasivo y patrimonio (equivalente al total de activos) al 31 de diciembre "
            "de 2025 fue de M$ 49.859.964, comparado con M$ 52.645.753 al cierre de 2024."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿A cuánto ascendieron las primas directas de CF Seguros de Vida en 2025?",
        "ground_truth": (
            "Las primas directas totales ascendieron a M$ 44.721.772 en 2025. Desglosadas por ramo: "
            "Vida hipotecario M$ 898.678, Adicionales M$ 652.172, Salud M$ 4.606.465, "
            "Accidentes M$ 134, y Vida otros (ramos masivos) M$ 38.564.323. "
            "La prima cedida por reaseguro fue de M$ 2.927.767, resultando en prima retenida "
            "de M$ 41.794.005."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    # ── Inversiones ───────────────────────────────────────────────────────────
    {
        "question": "¿Cuál es la composición de la cartera de inversiones de renta fija de CF Seguros de Vida al 31 de diciembre de 2025?",
        "ground_truth": (
            "La cartera de renta fija nacional totalizó M$ 36.923.769 (100% del total de inversiones), "
            "compuesta por instrumentos del Estado por M$ 22.999.824 (69%) e instrumentos emitidos "
            "por el sistema financiero por M$ 13.923.945 (31%). No había inversiones en renta fija "
            "extranjera. No existían instrumentos en mora ni con deterioro en la cartera al cierre."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuál fue la clasificación de riesgo crediticio de la cartera de inversiones de CF Seguros de Vida en 2025?",
        "ground_truth": (
            "Al 31 de diciembre de 2025, el 96,47% de la cartera de inversiones (M$ 35.619.421) "
            "contaba con clasificación AAA, el 1,37% (M$ 505.400) con AA+, y el 2,16% "
            "(M$ 798.948) con AA. No había instrumentos clasificados AA-, BBB, BB o menor, "
            "ni sin clasificación. La política exige al menos 80% en instrumentos sobre AA-."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuál fue el movimiento de la cartera de inversiones de CF Seguros de Vida durante 2025?",
        "ground_truth": (
            "La cartera inició el año con un saldo de M$ 30.445.226. Durante 2025 se realizaron "
            "adiciones por M$ 339.763.779 y ventas por M$ -275.695.770. Los vencimientos ascendieron "
            "a M$ -64.830.742. El devengo de intereses fue M$ 1.321.449, la utilidad por unidad "
            "reajustable M$ 857.602, y hubo una reclasificación de M$ 2.828.904 correspondiente a "
            "depósitos menores de 90 días. El saldo final fue M$ 34.730.721."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },
    # ── Riesgo ────────────────────────────────────────────────────────────────
    {
        "question": "¿Cuáles son los principales riesgos identificados y cómo los gestiona CF Seguros de Vida?",
        "ground_truth": (
            "La Compañía ha implementado un Sistema de Gestión de Riesgos (SGR) estructurado en "
            "dos componentes: la Estrategia de Gestión de Riesgos (que define el apetito de riesgos, "
            "roles y mecanismos de control) y las Políticas Específicas (lineamientos del Directorio "
            "para cada tipo de riesgo). El principal riesgo identificado es el Riesgo de Crédito: "
            "posibilidad de incumplimiento de contrapartes. Se gestiona con monitoreo periódico de "
            "clasificaciones de riesgo y exposición, reportes al Comité de Riesgos y Seguridad, "
            "y una Política de Riesgo de Crédito alineada con la NCG 325 de la CMF. El sistema "
            "se sustenta en ISO 31.000, directrices de Basilea y NCG de la CMF."
        ),
        "category": "riesgo",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuáles son las restricciones de política de inversión aplicadas al riesgo de crédito en CF Seguros de Vida?",
        "ground_truth": (
            "Las restricciones establecidas son: (1) al menos el 80% de la cartera debe estar en "
            "instrumentos con clasificación sobre AA-, y el 20% restante puede estar en A y A+; "
            "(2) máxima concentración por sector económico en renta fija local de 40%, excepto "
            "Bancos con límite de 80% y sector estatal con límite de 100%; "
            "(3) plazo máximo al vencimiento de cada instrumento de 5 años."
        ),
        "category": "riesgo",
        "filter": {"year": "2025"}
    },
    # ── Reaseguro ─────────────────────────────────────────────────────────────
    {
        "question": "¿Quién es el reasegurador de CF Seguros de Vida y cuál es su clasificación de riesgo?",
        "ground_truth": (
            "El reasegurador de CF Seguros de Vida es BNP Paribas Cardif Seguros de Vida S.A. "
            "(RUT 96.837.630-6), con sede en Chile. Cuenta con clasificación de riesgo AA (Estable) "
            "otorgada por ICR (fecha 17-01-2025) y FR (fecha 07-02-2025). "
            "Al 31 de diciembre de 2025 el saldo de siniestros por cobrar a este reasegurador "
            "era de M$ 43.592 (en moneda nacional, cobrable en enero 2026)."
        ),
        "category": "riesgo",
        "filter": {"year": "2025"}
    },
    # ── Gobernanza ────────────────────────────────────────────────────────────
    {
        "question": "¿Cuál es la estructura organizacional de CF Seguros de Vida?",
        "ground_truth": (
            "La estructura organizacional de CF Seguros de Vida, al 2025, es la siguiente: "
            "en la cúpula se encuentra el Directorio, seguido de Auditoría Interna y una Secretaria. "
            "Reportan al Directorio: el Gerente General, el Abogado Jefe, el Gerente de Admin. y "
            "Finanzas, el Gerente de Operaciones y Sistemas, el Jefe Técnico, el Jefe de Control "
            "Permanente y Riesgos, y el Jefe de Productos. Bajo el Gerente de Admin. y Finanzas "
            "se encuentran: Jefe de Finanzas (con Analista de Finanzas), Analista Técnico e "
            "Inversiones Senior, Analista Actuarial y Abogado. Bajo el Gerente de Operaciones "
            "y Sistemas: Jefe de Operaciones (con Analista de Operaciones y Analista de "
            "Operaciones Senior), Desarrollador Backend y Security Risk Specialist."
        ),
        "category": "gobernanza",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuáles son las responsabilidades en materia de sostenibilidad, comunicaciones y gestión de riesgos en CF Seguros de Vida?",
        "ground_truth": (
            "Según la Memoria 2025: el área de Recursos Humanos de Falabella Financiero es "
            "responsable de gestionar los temas de sostenibilidad y liderazgo al interior de "
            "la Compañía. El Gerente General de CF Seguros de Vida, junto con el área de "
            "Comunicaciones Corporativas, son los responsables de las relaciones con accionistas, "
            "inversionistas y medios de prensa. El Jefe de Control Permanente y Riesgos es quien "
            "está a cargo de la gestión de riesgos."
        ),
        "category": "gobernanza",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Qué establece la Política de Equidad y cuál es la meta de participación femenina en CF Seguros de Vida?",
        "ground_truth": (
            "La Política de Equidad (NCG 461-5.4.1) define cuatro principios fundamentales de "
            "compensación: (1) Igualdad de oportunidades, sin distinción de género u origen; "
            "(2) Competitividad con el mercado, mediante comparaciones periódicas; "
            "(3) Equidad salarial basada en mérito, experiencia y responsabilidades del cargo; "
            "(4) Reconocimiento del desempeño destacado. La meta de participación de género "
            "establecida es mantener una participación femenina del 40% en la fuerza laboral."
        ),
        "category": "gobernanza",
        "filter": {"year": "2025"}
    },
    # ── Sostenibilidad / Proveedores ──────────────────────────────────────────
    {
        "question": "¿Cuántos proveedores evaluó CF Seguros de Vida en 2025 y en qué materias?",
        "ground_truth": (
            "En 2025 CF Seguros de Vida evaluó un total de 86 proveedores: 83 nacionales y 3 "
            "extranjeros. El 100% de las evaluaciones incluyó temas de sostenibilidad (ASG). "
            "El porcentaje de proveedores significativos evaluados fue del 100% en el segmento "
            "nacional. La Compañía no posee acuerdos con pagos excepcionales con ningún prestador "
            "de servicio y no se registraron pagos de intereses por mora en pago de facturas."
        ),
        "category": "sostenibilidad",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Qué sanciones recibió CF Seguros de Vida en 2025 en materia laboral y de protección al cliente?",
        "ground_truth": (
            "Durante el año 2025 CF Seguros de Vida no recibió sanciones ni montos asociados "
            "en materia laboral (NCG 461-8.2). Tampoco se registraron sanciones cursadas por "
            "la autoridad en materia de protección de derechos del cliente (NCG 461-8.1)."
        ),
        "category": "sostenibilidad",
        "filter": {"year": "2025"}
    },
    # ── Prevención de delitos / Capacitación ──────────────────────────────────
    {
        "question": "¿Qué ilícitos prohíbe el Modelo de Prevención de Delitos de CF Seguros de Vida?",
        "ground_truth": (
            "El Modelo de Prevención de Delitos y la Política de Prevención de Delitos y Antisoborno "
            "prohíben categóricamente: cohecho, lavado de activos, financiamiento del terrorismo, "
            "corrupción entre particulares y negociación incompatible. Estas políticas establecen "
            "los estándares de comportamiento para prevenir conductas que puedan generar "
            "responsabilidad penal a la compañía (NCG 461-3.6. xiii, 8.5)."
        ),
        "category": "gobernanza",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Qué incluye el programa de divulgación y capacitación anual de CF Seguros de Vida?",
        "ground_truth": (
            "El programa incluye dos tipos de formación: (1) Cursos Normativos: programa anual "
            "para todos los colaboradores en Seguridad de la Información, Riesgo Operacional, "
            "Continuidad del Negocio, Prevención de Riesgos, Prevención del Delito, Libre "
            "Competencia, Mi Cliente y Datos Personales. (2) Capacitaciones impartidas por "
            "líderes de área y programas del Corporativo Falabella, incluyendo: Política de "
            "Seguridad, Yo Juego Limpio, Guía Legal, Prevención de Lavado de Activos y Combate "
            "al Financiamiento del Terrorismo, Programa de Ética, Riesgo Operacional, Evaluación "
            "de Proyectos, Libre Competencia, Datos Personales, Primeros Auxilios y Prevención "
            "de Riesgos."
        ),
        "category": "gobernanza",
        "filter": {"year": "2025"}
    },
    # ── Normativa contable ────────────────────────────────────────────────────
    {
        "question": "¿Bajo qué marco normativo contable fueron preparados los estados financieros de CF Seguros de Vida 2025?",
        "ground_truth": (
            "Los estados financieros fueron preparados de acuerdo con las Normas e Instrucciones "
            "impartidas por la Comisión para el Mercado Financiero (CMF). En los casos no normados "
            "por el regulador, se aplican las Normas Internacionales de Información Financiera "
            "(NIIF). Al 31 de diciembre de 2025 no existen ajustes a períodos anteriores ni "
            "cambios contables, ni reclasificaciones. La Compañía estima que no existen indicios "
            "que afecten la hipótesis de empresa en marcha."
        ),
        "category": "financiero_narrativa",
        "filter": {"year": "2025"}
    },
    {
        "question": "¿Cuáles son las deudas con intermediarios que reporta CF Seguros de Vida al 31 de diciembre de 2025?",
        "ground_truth": (
            "Las deudas con intermediarios al 31 de diciembre de 2025 totalizan M$ 445.247, "
            "correspondientes íntegramente a comisiones de intermediación con corredores de "
            "seguros relacionados: Seguros Falabella Corredores Ltda. y BancoFalabella "
            "Corredores de Seguros Ltda. Todos son pasivos corrientes (corto plazo). "
            "No hay deudas con asesores previsionales ni con terceros no relacionados."
        ),
        "category": "financiero",
        "filter": {"year": "2025"}
    },






    # ── Financiero: numérico ───────────────────────────────────────────────────
    {
        "question": "¿Cuáles fueron los ingresos por inversiones reportados por CF Seguros de Vida al 31 de diciembre de 2024?",
        "ground_truth": (
            "El resultado total de inversiones al 31 de diciembre de 2024 fue de M$1.144.927. "
            "De ese total, M$1.108.817 provienen de inversiones en renta fija (estatales M$755.177 "
            "y bancarios M$353.640) y M$36.110 de renta variable (fondos mutuos). "
            "El monto total invertido ascendió a M$36.846.345."
        ),
        "category": "financiero",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Cuál es el valor de mercado de la cartera de inversiones de renta fija de CF Seguros de Vida al cierre de 2024?",
        "ground_truth": (
            "Al 31 de diciembre de 2024, la valorización a mercado de la cartera de renta fija "
            "fue de M$35.467.180. La valorización de compra correspondiente era M$35.365.276, "
            "generando un mayor valor de mercado de M$101.904."
        ),
        "category": "financiero",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Cuánto fue el costo de siniestros del ejercicio 2024 en CF Seguros de Vida?",
        "ground_truth": (
            "El costo de siniestros del ejercicio 2024 ascendió a M$6.232.964. "
            "De ese total, M$4.760.741 corresponden a siniestros pagados y M$1.472.223 "
            "a variación de reserva de siniestros. La reserva de siniestros al cierre "
            "del ejercicio fue de M$8.821.340."
        ),
        "category": "financiero",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿A cuánto ascendió la inversión total de CF Seguros de Vida en capacitación durante 2024?",
        "ground_truth": (
            "El monto total invertido en capacitación durante 2024 fue de $8.322.881, "
            "lo que representa un 0,026% de los ingresos de la Compañía. "
            "Durante ese período, 21 personas se capacitaron, equivalente al 100% de la dotación."
        ),
        "category": "social",
        "filter": {"year": "2024"}
    },

    # ── Ambiental ─────────────────────────────────────────────────────────────
    {
        "question": "¿Cuántas emisiones de gases de efecto invernadero (Scope 1 y 2) reportó CF Seguros de Vida en su Memoria 2024?",
        "ground_truth": (
            "La Memoria Anual 2024 de CF Seguros de Vida no reporta métricas de emisiones "
            "de gases de efecto invernadero Scope 1 ni Scope 2. El documento no incluye "
            "un capítulo de huella de carbono ni indicadores ambientales cuantitativos."
        ),
        "category": "ambiental",
        "filter": {"year": "2024"}
    },

    # ── Riesgo ────────────────────────────────────────────────────────────────
    {
        "question": "¿Cuáles son los principales riesgos de negocio y riesgos de siniestralidad de CF Seguros de Vida?",
        "ground_truth": (
            "Los principales riesgos identificados en la Nota 6 son: (1) Riesgo de mercado, "
            "asociado a variaciones en tasas de interés sobre la cartera de renta fija — "
            "un alza de 200 pb impactaría el patrimonio en M$(887.531); (2) Riesgo de grupo, "
            "vinculado a pérdidas por contagio y riesgo reputacional derivados de problemas "
            "del Grupo Controlador Falabella; (3) Riesgo de siniestralidad, gestionado "
            "mediante provisiones técnicas que al 31/12/2024 totalizaron M$8.821.340 en "
            "reserva de siniestros, incluyendo M$2.085.295 por siniestros ocurridos y no "
            "reportados (OYNR). La Compañía no mantiene productos derivados."
        ),
        "category": "riesgo",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Cuál es el impacto en el patrimonio de CF Seguros de Vida ante un alza de 100 puntos base en tasas de interés?",
        "ground_truth": (
            "Ante un alza de 100 puntos base en las tasas de interés, el impacto proyectado "
            "en el patrimonio de CF Seguros de Vida sería de M$(450.639), reduciendo el "
            "patrimonio neto de M$25.113.651 (base) a M$24.663.012. El indicador "
            "P. neto / P. riesgo pasaría de 7,26 a 7,13."
        ),
        "category": "riesgo",
        "filter": {"year": "2024"}
    },

    # ── Financiero narrativo / Inversiones ────────────────────────────────────
    {
        "question": "¿Qué información reporta CF Seguros de Vida sobre su política y foco de inversiones de renta fija?",
        "ground_truth": (
            "La cartera de inversiones se concentra íntegramente en renta fija nacional "
            "(M$35.467.180 de un total de M$36.846.345). Dentro de renta fija, "
            "M$24.544.589 corresponden a instrumentos estatales y M$10.922.591 a bancarios. "
            "La Compañía no posee inversiones corporativas, securitizadas, mutuos hipotecarios, "
            "ni inversiones en el extranjero. Mensualmente se monitorean las diferencias entre "
            "valorización de compra y valorización de mercado. Los ingresos de activos a costo "
            "amortizado se reconocen en el Estado de Resultado Integral, aunque al 31/12/2024 "
            "no se registran inversiones bajo esa modalidad."
        ),
        "category": "financiero_narrativa",
        "filter": {"year": "2024"}
    },

    # ── Gobernanza ────────────────────────────────────────────────────────────
    {
        "question": "¿Cuáles son las responsabilidades del Directorio sobre la gestión de riesgos en CF Seguros de Vida?",
        "ground_truth": (
            "El Directorio se reúne periódicamente con las unidades de Riesgo, Auditoría Interna, "
            "Auditoría Externa y Sostenibilidad. Con el área de Riesgos se realizan dos "
            "presentaciones anuales con indicadores de riesgo contingentes. Con Auditoría Interna "
            "se reúne al menos una vez al año para monitorear el plan de auditoría y aprobar "
            "presupuesto; con Auditoría Externa la periodicidad es cuatrimestral. Además, "
            "cualquier utilización de productos derivados debe ser aprobada por el Directorio "
            "a través del Comité de Riesgos y Seguridad, quien define las políticas, "
            "procedimientos y mecanismos de control asociados."
        ),
        "category": "gobernanza",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Quiénes son los accionistas de CF Seguros de Vida y en qué porcentaje participan?",
        "ground_truth": (
            "Los accionistas de CF Seguros de Vida S.A. son: Falabella Inversiones Financieras S.A. "
            "(RUT 76.046.433-3) con un 94,2% de participación, y BNP Paribas Cardif Seguros de Vida S.A. "
            "(RUT 96.837.630-6) con un 5,8%. La controladora última del grupo es Falabella S.A."
        ),
        "category": "gobernanza",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Qué clasificación de riesgo tiene CF Seguros de Vida según las clasificadoras locales en 2024?",
        "ground_truth": (
            "CF Seguros de Vida cuenta con clasificación AA- Estables otorgada por dos clasificadoras: "
            "Fitch Chile Clasificadora de Riesgo (clasificación del 04-10-2024) y "
            "Feller-Rate Clasificadora de Riesgo (clasificación del 08-11-2024)."
        ),
        "category": "gobernanza",
        "filter": {"year": "2024"}
    },

    # ── Política contable / Siniestros ────────────────────────────────────────
    {
        "question": "¿Cómo reconoce CF Seguros de Vida el costo de siniestros en sus estados financieros?",
        "ground_truth": (
            "El costo estimado de siniestros se reconoce en función de la fecha de ocurrencia, "
            "registrando en el Estado de Resultado Integral todos los gastos hasta la liquidación. "
            "Para siniestros ocurridos pero no comunicados al cierre, se estima el costo con base "
            "en experiencia histórica mediante la provisión para prestaciones pendientes de "
            "declaración (OYNR). Los pagos se realizan con cargo a la provisión previa. "
            "Los siniestros denunciados se provisionan según condiciones de póliza; si no hay "
            "información suficiente, se provisionan estimativamente hasta conocer el monto exacto."
        ),
        "category": "riesgo",
        "filter": {"year": "2024"}
    },
    {
        "question": "¿Cuántos trabajadores registra CF Seguros de Vida y cuál es su dotación total?",
        "ground_truth": (
            "CF Seguros de Vida registra 21 trabajadores al 31 de diciembre de 2024, "
            "lo que constituye el 100% de su dotación. Durante 2024, las 21 personas "
            "se capacitaron a través de algún programa o curso."
        ),
        "category": "social",
        "filter": {"year": "2024"}
    },
]



# Instanciar métricas con Gemini nativo (ragas 0.4.3)
# Las nuevas métricas se puntúan por fila con metric.score(), no con ragas.evaluate()
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _build_metrics():
    return {
        "faithfulness":       Faithfulness(llm=_ragas_llm),
        "answer_relevancy":   AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings),
        "context_precision":  ContextPrecisionWithReference(llm=_ragas_llm),
        "context_recall":     ContextRecall(llm=_ragas_llm),
    }


def evaluate_mode(rag: MemoriasRAG, questions: list[dict]) -> pd.DataFrame:
    """Corre métricas RAGAS 0.4.x sobre un modo y retorna un DataFrame con los resultados."""
    rows = []

    print(f"  Generando y evaluando {len(questions)} preguntas en modo {rag.mode.upper()}...")
    metrics = _build_metrics()

    for i, item in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {item['question'][:70]}...")
        result = rag.answer(
            item["question"],
            filters=item.get("filter"),
        )
        q   = item["question"]
        a   = result["answer"]
        ctx = [s["text"] for s in result["sources"]]
        gt  = item["ground_truth"]

        scores = {}
        for name, metric in metrics.items():
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    if name == "faithfulness":
                        r = metric.score(user_input=q, response=a, retrieved_contexts=ctx)
                    elif name == "answer_relevancy":
                        r = metric.score(user_input=q, response=a)
                    elif name == "context_precision":
                        r = metric.score(user_input=q, reference=gt, retrieved_contexts=ctx)
                    elif name == "context_recall":
                        r = metric.score(user_input=q, retrieved_contexts=ctx, reference=gt)
                    scores[name] = float(r.value)
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"      ⚠️  {name}: {e} — retry en {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"      ❌  {name}: {e}")
                        scores[name] = float("nan")

        rows.append({
            "question":     q,
            "answer":       a,
            "contexts":     ctx,
            "ground_truth": gt,
            "category":     item.get("category", "general"),
            "mode":         rag.mode,
            **scores,
        })

    return pd.DataFrame(rows)


def run_comparative_evaluation() -> None:
    print("=" * 65)
    print("EVALUACIÓN COMPARATIVA RAGAS")
    print("Modo A (simple) vs Modo B (enriched)")
    print("=" * 65)

    results_by_mode = {}

    # ── Modo A ────────────────────────────────────────────────────────────────
    print("\n🔵 Modo A: SIMPLE (solo MarkItDown)...")
    try:
        df_simple = evaluate_mode(MemoriasRAG(mode="simple"), TEST_QUESTIONS)
        results_by_mode["simple"] = df_simple
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Asegúrate de haber indexado con USE_LANGEXTRACT=False.")

    # ── Modo B ────────────────────────────────────────────────────────────────
    print("\n🟢 Modo B: ENRIQUECIDO (MarkItDown + LangExtract)...")
    try:
        df_enriched = evaluate_mode(MemoriasRAG(mode="enriched"), TEST_QUESTIONS)
        results_by_mode["enriched"] = df_enriched
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Asegúrate de haber ejecutado extract_structured.py e indexado con USE_LANGEXTRACT=True.")

    if not results_by_mode:
        print("\n❌ Sin resultados para comparar.")
        return

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTADOS COMPARATIVOS — PROMEDIOS")
    print("=" * 65)

    summary_rows = []
    for mode, df in results_by_mode.items():
        row = {"modo": mode}
        for metric in METRIC_NAMES:
            if metric in df.columns:
                row[metric] = round(df[metric].mean(), 3)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("modo")
    print(summary_df.to_string())

    # ── Delta entre modos ─────────────────────────────────────────────────────
    if "simple" in results_by_mode and "enriched" in results_by_mode:
        print("\n📊 Delta (enriquecido − simple):")
        deltas = []
        for metric in METRIC_NAMES:
            if metric in summary_df.columns and len(summary_df) == 2:
                delta = summary_df.loc["enriched", metric] - summary_df.loc["simple", metric]
                deltas.append(delta)
                icon = "🟢 +" if delta > 0.01 else "🔴 " if delta < -0.01 else "⚪  "
                print(f"   {icon}{metric:<25} {delta:+.3f}")

        print("\n💡 Conclusión:")
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        if avg_delta > 0.05:
            print("   → LangExtract mejora significativamente las métricas.")
            print("     El costo adicional se justifica para este corpus.")
        elif avg_delta > 0.01:
            print("   → LangExtract produce una mejora modesta.")
            print("     Evalúa si el costo/tiempo vale para tu caso de uso.")
        else:
            print("   → LangExtract no produce mejora significativa.")
            print("     El Modo A (solo MarkItDown) es suficiente.")

        # Análisis por categoría de pregunta
        print("\n📊 Análisis por categoría de pregunta:")
        all_df = pd.concat(results_by_mode.values(), ignore_index=True)
        cat_pivot = all_df.groupby(["mode", "category"])[METRIC_NAMES].mean().round(3)
        print(cat_pivot.to_string())

    # ── Guardar resultados ────────────────────────────────────────────────────
    os.makedirs("evaluation", exist_ok=True)
    summary_df.to_csv("evaluation/comparison_summary.csv")
    all_results = pd.concat(results_by_mode.values(), ignore_index=True)
    all_results.to_csv("evaluation/results_detailed.csv", index=False)
    if "simple" in results_by_mode and "enriched" in results_by_mode:
        cat_pivot.to_csv("evaluation/results_by_category.csv")

    print("\nArchivos guardados:")
    print("  evaluation/comparison_summary.csv  — resumen por modo")
    print("  evaluation/results_detailed.csv    — fila por pregunta")
    print("  evaluation/results_by_category.csv — desglose por categoría")


if __name__ == "__main__":
    run_comparative_evaluation()

    # Cerrar sesión aiohttp del cliente google-genai para evitar warning al salir
    import asyncio, gc
    gc.collect()
    try:
        loop = asyncio.new_event_loop()
        sess = _genai_client._api_client._httpx_client  # type: ignore
    except Exception:
        pass
