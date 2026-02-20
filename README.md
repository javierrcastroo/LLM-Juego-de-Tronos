# LLM-Juego-de-Tronos

Pipeline RAG para responder preguntas sobre **Juego de Tronos (Libro 1)** y generar imágenes de escenas con modelos de Hugging Face.

## Modelos
- **LLM QA + planner + verificador**: `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Embeddings**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-large`
- **Imagen**: `stabilityai/stable-diffusion-3.5-large`
- **Selección automática de mejor imagen**: `openai/clip-vit-large-patch14`

## Estado actual
- `llm.ipynb` es **autocontenido** (no depende del `.py`).
- `got_rag_image_pipeline.py` queda como módulo alternativo/referencia.

## Mejoras “a lo bestia” incluidas en el notebook
1. **Retrieval híbrido**: dense (FAISS) + lexical (BM25) con fusión de scores.
2. **Expansión de consulta** (multi-query) opcional con LLM.
3. **Diversificación MMR** para evitar chunks redundantes.
4. **Respuesta grounded** con formato de evidencias y referencias.
5. **Verificación de fidelidad** de la respuesta y reescritura automática si detecta problemas.
6. **Planner de escena robusto** (parser JSON tolerante + retry fixer + fallback determinista).
7. **Prompt visual avanzado** (estilo, atmósfera, paleta, cámara, props, negatives fuertes).
8. **Generación multi-seed** y selección de la mejor imagen por **score CLIP texto-imagen**.

## Flujo
EPUB -> capítulos -> chunks -> embeddings/BM25 -> retrieval híbrido + rerank + MMR -> QA -> verificación -> planner escena -> prompt -> SD3.5 multi-seed -> ranking CLIP -> imagen final.

## Nota de hardware
Para correr completo en Colab, se recomienda GPU potente y buena VRAM.
