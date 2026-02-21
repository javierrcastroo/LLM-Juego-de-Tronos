from __future__ import annotations

import re
from dataclasses import dataclass

CHARACTERS = {
    "Eddard Stark": [r"\beddard\b", r"\bned\b"],
    "Catelyn Stark": [r"\bcatelyn\b"],
    "Robb Stark": [r"\brobb\b"],
    "Sansa Stark": [r"\bsansa\b"],
    "Arya Stark": [r"\barya\b"],
    "Bran Stark": [r"\bbran\b"],
    "Rickon Stark": [r"\brickon\b"],
    "Jon Snow": [r"\bjon snow\b", r"\bjon\b"],
    "Benjen Stark": [r"\bbenjen\b"],
    "Robert Baratheon": [r"\brobert\b"],
    "Cersei Lannister": [r"\bcersei\b"],
    "Jaime Lannister": [r"\bjaime\b"],
    "Tyrion Lannister": [r"\btyrion\b"],
    "Daenerys Targaryen": [r"\bdaenerys\b", r"\bdany\b"],
    "Viserys Targaryen": [r"\bviserys\b"],
    "Khal Drogo": [r"\bdrogo\b"],
    "Jorah Mormont": [r"\bjorah\b"],
}

HOUSES = {
    "House Stark": [r"\bstark\b"],
    "House Lannister": [r"\blannister\b"],
    "House Baratheon": [r"\bbaratheon\b"],
    "House Targaryen": [r"\btargaryen\b"],
    "Night's Watch": [r"\bnight'?s watch\b", r"\bmuro\b", r"\bthe wall\b"],
}

LOCATIONS = {
    "Winterfell": [r"\bwinterfell\b", r"\binvernalia\b"],
    "King's Landing": [r"\bking'?s landing\b", r"\bdesembarco del rey\b"],
    "The Wall": [r"\bthe wall\b", r"\bel muro\b"],
    "The North": [r"\bthe north\b", r"\belnorte\b"],
    "Pentos": [r"\bpentos\b"],
    "Dothraki Sea": [r"\bdothraki sea\b", r"\bmar dothraki\b"],
}


@dataclass
class ExtractedEntities:
    characters: list[str]
    houses: list[str]
    locations: list[str]


def _match_catalog(text: str, catalog: dict[str, list[str]]) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for canonical, patterns in catalog.items():
        if any(re.search(pat, low) for pat in patterns):
            found.append(canonical)
    return sorted(set(found))


def extract_entities(text: str) -> ExtractedEntities:
    return ExtractedEntities(
        characters=_match_catalog(text, CHARACTERS),
        houses=_match_catalog(text, HOUSES),
        locations=_match_catalog(text, LOCATIONS),
    )
