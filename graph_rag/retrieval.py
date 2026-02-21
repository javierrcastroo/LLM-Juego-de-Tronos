from __future__ import annotations

from dataclasses import dataclass

from neo4j import GraphDatabase

from graph_rag.config import Neo4jConfig


@dataclass
class GraphAnswer:
    question: str
    context: str
    answer: str


def retrieve_context(question: str, cfg: Neo4jConfig, top_k: int = 6) -> str:
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
    with driver.session(database=cfg.database) as session:
        result = session.run(
            """
            CALL db.index.fulltext.queryNodes('chunk_text', $question) YIELD node, score
            OPTIONAL MATCH (char:Character)-[:APPEARS_IN]->(node)
            OPTIONAL MATCH (house:House)-[:MENTIONED_IN]->(node)
            OPTIONAL MATCH (loc:Location)-[:MENTIONED_IN]->(node)
            RETURN node.chunk_id AS chunk_id,
                   left(node.text, 900) AS text,
                   score,
                   collect(DISTINCT char.name) AS characters,
                   collect(DISTINCT house.name) AS houses,
                   collect(DISTINCT loc.name) AS locations
            ORDER BY score DESC
            LIMIT $top_k
            """,
            question=question,
            top_k=top_k,
        )
        rows = list(result)
    driver.close()

    blocks = []
    for row in rows:
        meta = []
        if row["characters"]:
            meta.append("chars=" + ", ".join([x for x in row["characters"] if x]))
        if row["houses"]:
            meta.append("houses=" + ", ".join([x for x in row["houses"] if x]))
        if row["locations"]:
            meta.append("locations=" + ", ".join([x for x in row["locations"] if x]))
        metadata = f" ({' | '.join(meta)})" if meta else ""
        blocks.append(f"[{row['chunk_id']}] {row['text']}{metadata}")

    return "\n\n".join(blocks)


def simple_graph_rag_answer(question: str, cfg: Neo4jConfig) -> GraphAnswer:
    context = retrieve_context(question, cfg)
    if not context.strip():
        answer = "No encontré contexto en el grafo para responder esta pregunta."
    else:
        answer = (
            "Respuesta basada en Graph RAG (borrador):\n"
            f"Pregunta: {question}\n"
            "Revisa los fragmentos y entidades enlazadas para redactar la respuesta final:\n\n"
            f"{context}"
        )
    return GraphAnswer(question=question, context=context, answer=answer)
