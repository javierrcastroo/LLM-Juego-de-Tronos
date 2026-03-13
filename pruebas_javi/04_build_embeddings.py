import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading embedding model:", args.model)
    model = SentenceTransformer(args.model)

    texts = []
    items = []

    with open(args.chunks, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"].strip()
            if not text:
                continue

            texts.append("passage: " + text)
            items.append(obj)

    print("Encoding {} chunks...".format(len(texts)))
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, args.index)

    payload = {
        "embedding_model": args.model,
        "faiss_index_type": "IndexFlatIP",
        "normalize_embeddings": True,
        "items": items
    }

    with open(args.meta, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print("indexed chunks:", len(items))
    print("saved index:", args.index)
    print("saved meta:", args.meta)


if __name__ == "__main__":
    main()