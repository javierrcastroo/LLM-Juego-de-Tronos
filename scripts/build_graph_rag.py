from __future__ import annotations

import argparse

from graph_rag.config import get_neo4j_config
from graph_rag.ingest import ingest_epub_to_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta EPUB -> Neo4j para Graph RAG")
    parser.add_argument("--epub", default="Canci_243_n_de_Hielo_y_Fuego_01_-_Juego_de_Tronos.epub")
    args = parser.parse_args()

    cfg = get_neo4j_config()
    result = ingest_epub_to_graph(args.epub, cfg)

    print("Ingesta finalizada")
    print(result)


if __name__ == "__main__":
    main()
