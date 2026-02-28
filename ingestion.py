"""Colab-oriented ingestion script for persistent ChromaDB knowledge base.

Target persistence path:
/content/drive/MyDrive/NLP/PROYECTO/rag_knowledge_base
"""

from pathlib import Path

PERSIST_BASE = Path('/content/drive/MyDrive/NLP/PROYECTO/rag_knowledge_base')

if __name__ == '__main__':
    print('Este proyecto usa Ingestion_Knowledge.ipynb como notebook principal de ingesta en Colab.')
    print('Ruta persistente esperada:', PERSIST_BASE)
