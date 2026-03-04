"""
Interfaz gráfica del Chatbot RAG con selector de modo.
Ejecutar: streamlit run app.py
"""
import streamlit as st
from src.chain import MemoriasRAG

st.set_page_config(page_title="RAG Memorias CMF Chile", page_icon="🇨🇱", layout="wide")
st.title("🇨🇱 Chatbot — Memorias Anuales CMF Chile")
st.markdown("Consulta en lenguaje natural las **Memorias Anuales Integradas NCG 461** del IPSA.")


@st.cache_resource(show_spinner="Iniciando Modo Simple...")
def load_rag_simple():
    return MemoriasRAG(mode="simple")


@st.cache_resource(show_spinner="Iniciando Modo Enriquecido...")
def load_rag_enriched():
    return MemoriasRAG(mode="enriched")


with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio(
        "Modo de búsqueda",
        ["🔵 Simple (solo MarkItDown)", "🟢 Enriquecido (+ LangExtract)"],
        help="**Simple:** chunks del texto Markdown directo.\n\n**Enriquecido:** agrega chunks de entidades estructuradas (indicadores financieros y ASG/ESG).",
    )
    use_enriched = "Enriquecido" in modo

    st.divider()
    st.header("🔍 Filtros")
    empresa = st.selectbox("Empresa", ["CF Seguros de Vida", "Todas"])
    año = st.selectbox("Año", ["Todos", "2024", "2023", "2022"])
    n_chunks = st.slider("Fragmentos a recuperar", 3, 10, 7)

    st.divider()
    st.header("💡 Ejemplos")
    EJEMPLOS = [
        "¿Cuáles son los principales riesgos de negocio y riesgos de siniestralidad de CF Seguros de Vida?",
        "¿Cuáles son las políticas de administración de liquidez para CF Seguros de Vida?",
        "Resume las responsabilidades del Directorio sobre la gestión de riesgos en CF Seguros de Vida",
        "¿Cuáles fueron los ingresos por inversiones y primas emitidas reportados?",
        "¿Qué información reporta la compañía sobre su política y foco de inversiones de renta fija?",
    ]
    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, use_container_width=True, key=ejemplo):
            st.session_state["pregunta_input"] = ejemplo

pregunta = st.text_area(
    "Tu pregunta:",
    value=st.session_state.get("pregunta_input", ""),
    height=80,
    placeholder="Ej: ¿Cuáles son las políticas de liquidez para CF Seguros de Vida?",
)

if st.button("🔍 Consultar", type="primary") and pregunta.strip():
    rag = load_rag_enriched() if use_enriched else load_rag_simple()
    filters = {}
    if empresa != "Todas": filters["company"] = empresa
    if año != "Todos": filters["year"] = año

    # ── Phase 1: retrieval + reranking (blocking) — show spinner ─────────────
    with st.spinner("Recuperando y reordenando fragmentos relevantes..."):
        stream, resultado = rag.answer_stream(
            pregunta, n_results=n_chunks, filters=filters if filters else None
        )

    # ── Phase 2: streaming generation — live token rendering ─────────────────
    st.markdown("---")
    modo_badge = "🟢 Enriquecido" if use_enriched else "🔵 Simple"
    st.subheader(f"📝 Respuesta — {modo_badge}")
    st.write_stream(stream)   # streams tokens live as they arrive from Gemini

    # ── Phase 3: metrics + sources ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Fuentes consultadas", resultado["n_chunks"])
    with col2: st.metric("Modo", resultado["mode"])
    with col3:
        if resultado["sources"]:
            st.metric("Similitud máxima", f"{max(s['similarity'] for s in resultado['sources']):.4f}")

    with st.expander(f"📚 Ver {len(resultado['sources'])} fragmentos recuperados"):
        for i, source in enumerate(resultado["sources"], 1):
            m = source["metadata"]
            tipo = m.get("chunk_type", "narrative_md")
            sim = source["similarity"]
            badge = "📊 ESTRUCTURADO" if tipo == "structured_entity" else "📄 NARRATIVO"
            sim_icon = "🟢" if sim > 0.7 else "🟡" if sim > 0.5 else "🔴"
            pagina_val = m.get("page_num", -1)
            pagina_str = f" | **Pág:** {pagina_val}" if pagina_val != -1 else ""
            st.markdown(f"**Fuente {i}** {sim_icon} {badge} | **{m.get('company','?')}** {m.get('year','?')}{pagina_str} | `{sim}`")
            st.text_area(f"Contenido {i}", source["text"][:600] + ("..." if len(source["text"]) > 600 else ""), height=130, key=f"src_{i}", disabled=True)
            st.divider()

elif st.session_state.get("consultar_pressed") and not pregunta.strip():
    st.warning("Por favor escribe una pregunta.")
