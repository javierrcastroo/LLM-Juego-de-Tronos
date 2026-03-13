import argparse
import json
import re
from collections import OrderedDict

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


def normalize_space(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_html_to_text(html_bytes):
    soup = BeautifulSoup(html_bytes, "lxml")

    for tag in soup(["script", "style", "head", "title", "meta", "link", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    return text.strip()


def get_spine_docs(book):
    id_to_item = {}
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            id_to_item[item.get_id()] = item

    docs = []
    for spine_entry in book.spine:
        if isinstance(spine_entry, tuple):
            item_id = spine_entry[0]
        else:
            item_id = spine_entry

        if item_id == "nav":
            continue

        item = id_to_item.get(item_id)
        if item is not None:
            docs.append(item)

    return docs


def guess_title_from_text(text, fallback):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback

    first = lines[0]
    first_norm = normalize_space(first)

    # Typical chapter-like titles
    if len(first_norm) <= 80:
        return first_norm

    return fallback


def extract_sections(epub_path):
    book = epub.read_epub(epub_path)

    docs = get_spine_docs(book)
    sections = []

    for i, item in enumerate(docs):
        raw_text = clean_html_to_text(item.get_content())

        if len(raw_text) < 120:
            continue

        fallback_title = "section_{:04d}".format(i)
        title = guess_title_from_text(raw_text, fallback_title)

        href = getattr(item, "file_name", None)

        sections.append({
            "section_id": len(sections),
            "spine_order": i,
            "title": title,
            "href": href,
            "text": raw_text
        })

    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epub", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sections = extract_sections(args.epub)

    with open(args.out, "w", encoding="utf8") as f:
        for row in sections:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("sections:", len(sections))


if __name__ == "__main__":
    main()