# LLM-Juego-de-Tronos

Pipeline RAG para responder preguntas sobre **Juego de Tronos (Libro 1)** y generar una imagen de la escena usando modelos de Hugging Face.

## Modelos configurados
- **LLM (QA + planificación de escena)**: `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Embeddings**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-large`
- **Generación de imagen**: `stabilityai/stable-diffusion-3.5-large`

## Archivos clave
- `got_rag_image_pipeline.py`: lógica principal (ingesta EPUB, chunking, retrieval, respuesta, planificación de escena y render de imagen).
- `llm.ipynb`: notebook simplificado para ejecutar el flujo en Colab.

## Flujo
1. Extraer capítulos desde el EPUB.
2. Crear chunks y guardarlos en parquet.
3. Indexar con FAISS + reranking.
4. Responder pregunta con contexto recuperado.
5. Generar JSON de escena **genérico** (sin hardcode por tipo de pregunta).
6. Convertir ese JSON en prompt de imagen y renderizar con SD3.5.

## Nota de hardware
- Recomendado: GPU con bastante VRAM.
- Si no cabe en memoria, reduce tamaño de imagen o pasos de inferencia en `ask_and_draw`.
