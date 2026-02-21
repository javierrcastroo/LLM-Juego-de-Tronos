# LLM-Juego-de-Tronos

Pipeline RAG para responder preguntas sobre **Juego de Tronos (Libro 1)** y generar una imagen de la escena usando modelos de Hugging Face.

## Modelos configurados
- **LLM (QA + planificación de escena)**: `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Embeddings**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-large`
- **Generación de imagen**: `stabilityai/stable-diffusion-3.5-large`

## Archivos clave
- `got_rag_image_pipeline.py`: lógica principal (ingesta EPUB, chunking, retrieval, respuesta, planificación de escena y render de imagen).
- `llm.ipynb`: notebook autocontenido para ejecutar todo el flujo en Colab (sin depender de archivos `.py`).
- `docker-compose.neo4j.yml`: despliegue de Neo4j persistente para Graph RAG.
- `graph_rag/`: módulo base para ingesta de EPUB, extracción de entidades y consultas Graph RAG.
- `scripts/build_graph_rag.py`: ingesta al grafo desde EPUB.
- `scripts/query_graph_rag.py`: consulta rápida del grafo.

## Flujo (RAG con embeddings actual)
1. Extraer capítulos desde el EPUB.
2. Crear chunks y guardarlos en parquet.
3. Indexar con FAISS + reranking.
4. Responder pregunta con contexto recuperado.
5. Generar JSON de escena **genérico** (sin hardcode por tipo de pregunta), con recuperación robusta y fallback si el LLM no devuelve JSON perfecto.
6. Convertir ese JSON en prompt de imagen y renderizar con SD3.5.

## Nuevo flujo Graph RAG (Neo4j persistente)
1. Levantar Neo4j con Docker y volúmenes persistentes (`graph_rag_data/neo4j`).
2. Ingerir el EPUB a nodos de tipo `Document`, `Chapter`, `Chunk`, `Character`, `House`, `Location`.
3. Persistir también los documentos procesados en parquet (`graph_rag_data/docs/chapters.parquet` y `chunks.parquet`).
4. Recuperar contexto por índice full-text de chunks + metadatos de entidades conectadas.

### 1) Arrancar Neo4j persistente
```bash
cp .env.neo4j.example .env.neo4j
docker compose --env-file .env.neo4j -f docker-compose.neo4j.yml up -d
```

UI web: `http://localhost:7474`  
Bolt: `bolt://localhost:7687`

### 2) Instalar dependencias Graph RAG
```bash
pip install -r requirements-graph-rag.txt
```

### 3) Ingesta de Juego de Tronos al grafo
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password1234
python scripts/build_graph_rag.py --epub Canci_243_n_de_Hielo_y_Fuego_01_-_Juego_de_Tronos.epub
```

### 4) Consulta rápida Graph RAG
```bash
python scripts/query_graph_rag.py "¿Quién acompaña a Eddard Stark al sur?"
```

## Uso desde Google Colab (GPU para modelos)
Objetivo: usar Colab para inferencia (LLM/embeddings) pero conectando contra la misma BBDD Neo4j persistida.

1. En tu máquina/servidor deja Neo4j levantado con IP pública o tunel seguro (por ejemplo Cloudflare Tunnel/Tailscale).  
2. Habilita acceso al puerto Bolt (`7687`) desde Colab de forma segura (IP allowlist o túnel autenticado).  
3. En Colab:
```python
!git clone <TU_REPO>
%cd LLM-Juego-de-Tronos
!pip install -r requirements-graph-rag.txt

import os
os.environ["NEO4J_URI"] = "bolt://<HOST_O_TUNEL>:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "<PASSWORD>"

!python scripts/query_graph_rag.py "¿Qué relación hay entre Stark y Lannister al inicio?"
```

> Recomendación: no exponer Neo4j directamente a internet sin capa de seguridad. Usa túnel autenticado o VPN.

## Modelo básico de entidades implementado
Se incluye un extractor base (rule-based) para empezar rápido con Game of Thrones:
- **Personajes**: Eddard, Catelyn, Arya, Sansa, Jon Snow, Tyrion, Daenerys, etc.
- **Casas/facciones**: Stark, Lannister, Baratheon, Targaryen, Night's Watch.
- **Lugares**: Winterfell, King's Landing, The Wall, Pentos, etc.

Además se crea una relación de co-ocurrencia `INTERACTS_WITH` entre personajes que aparecen en el mismo chunk.

## Nota de hardware
- Recomendado: GPU con bastante VRAM.
- Si no cabe en memoria, reduce tamaño de imagen o pasos de inferencia en `ask_and_draw`.
