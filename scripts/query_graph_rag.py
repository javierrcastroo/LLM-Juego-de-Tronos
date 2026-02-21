from __future__ import annotations

import argparse

from graph_rag.config import get_neo4j_config
from graph_rag.retrieval import simple_graph_rag_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta básica Graph RAG en Neo4j")
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    cfg = get_neo4j_config()
    res = simple_graph_rag_answer(args.question, cfg)
    print(res.answer)


if __name__ == "__main__":
    main()
