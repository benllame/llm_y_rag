from src.chain import MemoriasRAG

rag = MemoriasRAG(mode="simple")
res = rag.answer("¿A cuánto ascendió el total de patrimonio de CF Seguros de Vida al cierre de 2025?", n_results=5)
print(res["answer"])
print("SOURCES:")
for s in res["sources"]:
    print(s["metadata"], s["similarity"])
