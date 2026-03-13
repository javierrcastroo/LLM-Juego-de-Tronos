import argparse
import json

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


def generate_answer(tokenizer, model, prompt, max_new_tokens=220):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    if full_text.startswith(prompt_text):
        answer = full_text[len(prompt_text):].strip()
    else:
        answer = full_text.strip()

    return answer


def build_prompt(question, ranked_items):
    context_blocks = []

    for i, item in enumerate(ranked_items, start=1):
        chunk = item["chunk"]
        header = "[Contexto {} | chunk_id={} | section_id={} | title={}]".format(
            i,
            chunk.get("chunk_id"),
            chunk.get("section_id"),
            chunk.get("title", "")
        )
        context_blocks.append(header + "\n" + chunk["text"])

    context = "\n\n".join(context_blocks)

    prompt = """Eres un asistente de preguntas y respuestas sobre un libro.
Debes responder usando SOLO el contexto proporcionado.
No inventes datos.
Si el contexto no basta, di exactamente:
No puedo responder con seguridad usando solo el contexto recuperado.

Reglas:
- Responde en español.
- Sé directo y preciso.
- Si hay una respuesta clara, dila de forma explícita.
- Si puedes, menciona el chunk_id como evidencia.
- No uses conocimiento externo.

Contexto:
{context}

Pregunta:
{question}

Respuesta:""".format(context=context, question=question)

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--question", required=True)

    parser.add_argument("--embed_model", default=None)
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-base")
    parser.add_argument("--generator_model", default="Qwen/Qwen2.5-3B-Instruct")

    parser.add_argument("--top_k_retrieval", type=int, default=20)
    parser.add_argument("--top_k_final", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=220)

    args = parser.parse_args()

    print("Loading FAISS index...")
    index = faiss.read_index(args.index)

    print("Loading metadata...")
    with open(args.meta, "r", encoding="utf8") as f:
        meta_payload = json.load(f)

    items = meta_payload["items"]
    embed_model_name = args.embed_model or meta_payload["embedding_model"]

    print("Loading embedding model:", embed_model_name)
    embed_model = SentenceTransformer(embed_model_name)

    print("Encoding query...")
    query_emb = embed_model.encode(
        ["query: " + args.question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    print("Initial retrieval...")
    scores, indices = index.search(query_emb, args.top_k_retrieval)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        candidates.append({
            "faiss_score": float(score),
            "chunk": items[idx]
        })

    print("Retrieved candidates:", len(candidates))

    print("Loading reranker:", args.reranker_model)
    reranker = CrossEncoder(args.reranker_model)

    pairs = [[args.question, c["chunk"]["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=True)

    for c, rr in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rr)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    final_items = ranked[:args.top_k_final]

    print("\n=== TOP CHUNKS ===\n")
    for i, item in enumerate(final_items, start=1):
        chunk = item["chunk"]
        print(
            "[{}] chunk_id={} section_id={} title={} faiss={:.4f} rerank={:.4f}".format(
                i,
                chunk.get("chunk_id"),
                chunk.get("section_id"),
                chunk.get("title", ""),
                item["faiss_score"],
                item["rerank_score"]
            )
        )
        print(chunk["text"][:700])
        print("\n" + "-" * 100 + "\n")

    prompt = build_prompt(args.question, final_items)

    print("Loading generator:", args.generator_model)
    tokenizer, model = load_generator(args.generator_model)

    print("Generating answer...\n")
    answer = generate_answer(
        tokenizer,
        model,
        prompt,
        max_new_tokens=args.max_new_tokens
    )

    print("=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()