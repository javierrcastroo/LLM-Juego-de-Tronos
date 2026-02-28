# LLM-Juego-de-Tronos

Pipeline RAG para responder preguntas sobre **Juego de Tronos (Libro 1)** y generar una imagen de la escena usando modelos de Hugging Face.

## Modelos configurados
- **LLM (QA + planificación de escena)**: `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Embeddings (pipeline principal)**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-large`
- **Generación de imagen**: `stabilityai/stable-diffusion-3.5-large`

## Archivos clave
- `got_rag_image_pipeline.py`: lógica principal (ingesta EPUB, chunking, retrieval, respuesta, planificación de escena y render de imagen).
- `llm.ipynb`: notebook autocontenido para ejecutar el flujo principal en Colab.
- `Ingestion_Knowledge.ipynb`: notebook exclusivo para Colab que clona el repo, procesa el corpus y persiste una base vectorial ChromaDB + modelo de embeddings en Google Drive.

## Flujo del notebook `Ingestion_Knowledge.ipynb` (Colab + ChromaDB)
1. Montar Google Drive.
2. Definir ruta persistente única en Drive:
   - `/content/drive/MyDrive/NLP/PROYECTO/rag_knowledge_base`
3. Clonar el repo dentro de esa ruta persistente.
4. Instalar dependencias para ingestión con ChromaDB.
5. Cargar el EPUB del proyecto.
6. Chunkear capítulos con solapamiento.
7. Crear embeddings con `Alibaba-NLP/gte-large-en-v1.5`.
8. Persistir en Drive:
   - Base de datos Chroma (`.../chroma_db`)
   - Modelo de embeddings (`.../embedding_model`)
   - Parquet de chunks (`.../chunks/chunks.parquet`)

## Ejecución recomendada
Abre y ejecuta `Ingestion_Knowledge.ipynb` en Google Colab, de arriba abajo, para dejar todo persistido en Drive.

## Nota de hardware
- Para ingesta masiva de embeddings es recomendable usar GPU en Colab.
- Si hay límites de RAM/VRAM, reduce `BATCH_SIZE` en el notebook.
