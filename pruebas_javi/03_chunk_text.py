import argparse
import json
import re


TARGET_TOKENS = 420
MIN_TOKENS = 180
OVERLAP_TOKENS = 80
MAX_SENT_TOKENS = 120


def approx_token_count(text):
    return len(text.split())


def normalize_space(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def split_sentences(text):
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r'(?<=[\.\!\?\…])\s+(?=[A-ZÁÉÍÓÚÜÑ"“\'¿¡\(0-9])', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def split_long_paragraph(paragraph):
    sents = split_sentences(paragraph)
    if not sents:
        return [paragraph]

    pieces = []
    current = []
    current_tokens = 0

    for sent in sents:
        n = approx_token_count(sent)
        if current and current_tokens + n > MAX_SENT_TOKENS:
            pieces.append(" ".join(current).strip())
            current = [sent]
            current_tokens = n
        else:
            current.append(sent)
            current_tokens += n

    if current:
        pieces.append(" ".join(current).strip())

    return pieces


def paragraph_units(text):
    paragraphs = split_paragraphs(text)
    units = []

    for p in paragraphs:
        n = approx_token_count(p)

        # If paragraph is already manageable, keep it whole
        if n <= MAX_SENT_TOKENS * 2:
            units.append(p)
        else:
            units.extend(split_long_paragraph(p))

    return [u for u in units if u.strip()]


def build_chunks_from_units(units):
    chunks = []
    current_units = []
    current_tokens = 0

    for unit in units:
        n = approx_token_count(unit)

        if current_units and current_tokens + n > TARGET_TOKENS:
            chunk_text = "\n\n".join(current_units).strip()
            chunks.append(chunk_text)

            # overlap from tail
            overlap_units = []
            overlap_tokens = 0

            for prev in reversed(current_units):
                prev_n = approx_token_count(prev)
                overlap_units.insert(0, prev)
                overlap_tokens += prev_n
                if overlap_tokens >= OVERLAP_TOKENS:
                    break

            current_units = overlap_units[:]
            current_tokens = sum(approx_token_count(x) for x in current_units)

        current_units.append(unit)
        current_tokens += n

    if current_units:
        chunk_text = "\n\n".join(current_units).strip()
        chunks.append(chunk_text)

    # Merge too-short final chunks backward if needed
    merged = []
    for chunk in chunks:
        n = approx_token_count(chunk)
        if merged and n < MIN_TOKENS:
            merged[-1] = merged[-1].rstrip() + "\n\n" + chunk.lstrip()
        else:
            merged.append(chunk)

    return merged


def build_chunks(text):
    units = paragraph_units(text)
    if not units:
        return []
    return build_chunks_from_units(units)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    total_chunks = 0

    with open(args.out, "w", encoding="utf8") as fout:
        with open(args.infile, "r", encoding="utf8") as fin:
            for line in fin:
                obj = json.loads(line)

                section_id = obj["section_id"]
                title = obj.get("title", "")
                text = obj["text"]

                chunks = build_chunks(text)

                for i, chunk_text in enumerate(chunks):
                    out_obj = {
                        "chunk_id": total_chunks,
                        "section_id": section_id,
                        "title": title,
                        "chunk_in_section": i,
                        "char_count": len(chunk_text),
                        "token_count": approx_token_count(chunk_text),
                        "text": chunk_text
                    }
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print("chunks created:", total_chunks)


if __name__ == "__main__":
    main()