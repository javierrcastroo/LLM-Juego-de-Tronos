from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

from graph_rag.config import Neo4jConfig
from graph_rag.entities import extract_entities


@dataclass
class GraphIngestResult:
    chapters: int
    chunks: int
    characters: int
    houses: int
    locations: int


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def list_xhtml_text_files(zf: zipfile.ZipFile) -> list[str]:
    candidates = [f for f in zf.namelist() if f.lower().endswith((".xhtml", ".html"))]
    preferred = [f for f in candidates if "/text/" in f.lower() or "/texto/" in f.lower()]
    return sorted(preferred if preferred else candidates)


def extract_chapters(epub_path: str) -> pd.DataFrame:
    chapters: list[dict] = []
    with zipfile.ZipFile(epub_path, "r") as zf:
        for file_name in list_xhtml_text_files(zf):
            raw = zf.read(file_name)
            soup = BeautifulSoup(raw, "lxml")
            text = clean_text(soup.get_text("\n"))
            if len(text) < 800:
                continue
            chapters.append(
                {
                    "chapter_id": len(chapters),
                    "epub_file": file_name,
                    "title": file_name.split("/")[-1],
                    "text": text,
                    "n_chars": len(text),
                }
            )
    return pd.DataFrame(chapters)


def chunk_text(text: str, chunk_size: int = 2500, overlap: int = 300) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        start += max(1, chunk_size - overlap)
    return chunks


def build_chunks(chapters_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, chapter in chapters_df.iterrows():
        for i, (start, end, chunk) in enumerate(chunk_text(chapter["text"])):
            rows.append(
                {
                    "chunk_id": f"{int(chapter['chapter_id'])}_{i}",
                    "chapter_id": int(chapter["chapter_id"]),
                    "epub_file": chapter["epub_file"],
                    "start_char": start,
                    "end_char": end,
                    "text": chunk,
                }
            )
    return pd.DataFrame(rows)


def save_docs(chapters_df: pd.DataFrame, chunks_df: pd.DataFrame, out_dir: str = "graph_rag_data/docs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    chapters_df.to_parquet(os.path.join(out_dir, "chapters.parquet"), index=False)
    chunks_df.to_parquet(os.path.join(out_dir, "chunks.parquet"), index=False)


def ingest_epub_to_graph(epub_path: str, cfg: Neo4jConfig) -> GraphIngestResult:
    chapters_df = extract_chapters(epub_path)
    chunks_df = build_chunks(chapters_df)
    save_docs(chapters_df, chunks_df)

    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
    with driver.session(database=cfg.database) as session:
        session.run("CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        session.run("CREATE CONSTRAINT char_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE")
        session.run("CREATE CONSTRAINT house_name IF NOT EXISTS FOR (h:House) REQUIRE h.name IS UNIQUE")
        session.run("CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")
        session.run("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")

        session.run("MERGE (d:Document {name: 'A Game of Thrones'})")

        for _, row in chapters_df.iterrows():
            session.run(
                """
                MERGE (ch:Chapter {chapter_id: $chapter_id})
                SET ch.epub_file = $epub_file,
                    ch.title = $title,
                    ch.n_chars = $n_chars
                WITH ch
                MATCH (d:Document {name: 'A Game of Thrones'})
                MERGE (d)-[:HAS_CHAPTER]->(ch)
                """,
                chapter_id=int(row["chapter_id"]),
                epub_file=row["epub_file"],
                title=row["title"],
                n_chars=int(row["n_chars"]),
            )

        characters, houses, locations = set(), set(), set()
        for _, row in chunks_df.iterrows():
            extracted = extract_entities(row["text"])
            characters.update(extracted.characters)
            houses.update(extracted.houses)
            locations.update(extracted.locations)

            session.run(
                """
                MATCH (ch:Chapter {chapter_id: $chapter_id})
                MERGE (ck:Chunk {chunk_id: $chunk_id})
                SET ck.text = $text,
                    ck.start_char = $start_char,
                    ck.end_char = $end_char,
                    ck.epub_file = $epub_file
                MERGE (ch)-[:HAS_CHUNK]->(ck)
                """,
                chapter_id=int(row["chapter_id"]),
                chunk_id=row["chunk_id"],
                text=row["text"],
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                epub_file=row["epub_file"],
            )

            for name in extracted.characters:
                session.run(
                    """
                    MERGE (c:Character {name: $name})
                    WITH c
                    MATCH (ck:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:APPEARS_IN]->(ck)
                    """,
                    name=name,
                    chunk_id=row["chunk_id"],
                )

            for name in extracted.houses:
                session.run(
                    """
                    MERGE (h:House {name: $name})
                    WITH h
                    MATCH (ck:Chunk {chunk_id: $chunk_id})
                    MERGE (h)-[:MENTIONED_IN]->(ck)
                    """,
                    name=name,
                    chunk_id=row["chunk_id"],
                )

            for name in extracted.locations:
                session.run(
                    """
                    MERGE (l:Location {name: $name})
                    WITH l
                    MATCH (ck:Chunk {chunk_id: $chunk_id})
                    MERGE (l)-[:MENTIONED_IN]->(ck)
                    """,
                    name=name,
                    chunk_id=row["chunk_id"],
                )

        # Relaciones simples de co-ocurrencia entre personajes
        session.run(
            """
            MATCH (c1:Character)-[:APPEARS_IN]->(ck:Chunk)<-[:APPEARS_IN]-(c2:Character)
            WHERE c1.name < c2.name
            MERGE (c1)-[r:INTERACTS_WITH]-(c2)
            ON CREATE SET r.weight = 1
            ON MATCH SET r.weight = r.weight + 1
            """
        )

    driver.close()
    return GraphIngestResult(
        chapters=len(chapters_df),
        chunks=len(chunks_df),
        characters=len(characters),
        houses=len(houses),
        locations=len(locations),
    )
