from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password1234")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


def get_neo4j_config() -> Neo4jConfig:
    return Neo4jConfig()
