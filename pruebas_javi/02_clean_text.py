import argparse
import json
import re


BAD_TITLE_PATTERNS = [
    r"^cover$",
    r"^title page$",
    r"^copyright$",
    r"^contents$",
    r"^toc$",
    r"^índice$",
    r"^portada$",
    r"^créditos$",
]


def normalize_space(text):
    text = text.replace("\xa0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "—")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_bad_title(title):
    t = title.strip().lower()
    for pat in BAD_TITLE_PATTERNS:
        if re.match(pat, t):
            return True
    return False


def looks_like_noise(text):
    t = text.lower()

    if len(t) < 200:
        return True

    if "calibre" in t and len(t) < 800:
        return True

    return False


def clean_text(text):
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []

    for line in lines:
        if not line:
            cleaned.append("")
            continue

        # remove repeated ornamental separators
        if re.fullmatch(r"[-=*~·•\s]{3,}", line):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = normalize_space(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    kept = 0

    with open(args.out, "w", encoding="utf8") as fout:
        with open(args.infile, "r", encoding="utf8") as fin:
            for line in fin:
                obj = json.loads(line)

                title = obj.get("title", "").strip()
                text = clean_text(obj.get("text", ""))

                if is_bad_title(title):
                    continue

                if looks_like_noise(text):
                    continue

                obj["title"] = title
                obj["text"] = text

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print("clean sections:", kept)


if __name__ == "__main__":
    main()