"""Pipeline RAG + generación de imagen para Juego de Tronos (Libro 1).

Soporta:
- QA: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
- Imagen: stabilityai/stable-diffusion-3.5-large
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from diffusers import StableDiffusion3Pipeline
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
EMBED_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-large"
SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


@dataclass
class LoadedModels:
    tokenizer: AutoTokenizer
    llm: AutoModelForCausalLM
    embedder: SentenceTransformer
    reranker: CrossEncoder
    image_pipe: StableDiffusion3Pipeline


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_title_and_pov(text: str) -> tuple[Optional[str], Optional[str]]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:20]:
        if re.match(r"^[A-ZÁÉÍÓÚÑÜ]+.*\(\d+\)$", line):
            pov = line.split("(")[0].strip()
            return line, pov
    return None, None


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
            title, pov = extract_title_and_pov(text)
            chapters.append(
                {
                    "chapter_id": len(chapters),
                    "epub_file": file_name,
                    "title": title,
                    "pov": pov,
                    "text": text,
                    "n_chars": len(text),
                }
            )
    return pd.DataFrame(chapters)


def chunk_text(text: str, chunk_size: int = 4500, overlap: int = 750) -> list[tuple[int, int, str]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        start += chunk_size - overlap
    return chunks


def build_chunks(chapters_df: pd.DataFrame, chunk_size: int = 4500, overlap: int = 750) -> pd.DataFrame:
    rows = []
    for _, chapter in chapters_df.iterrows():
        for i, (start, end, chunk) in enumerate(chunk_text(chapter["text"], chunk_size, overlap)):
            rows.append(
                {
                    "chunk_id": f"{int(chapter['chapter_id'])}_{i}",
                    "chapter_id": int(chapter["chapter_id"]),
                    "epub_file": chapter["epub_file"],
                    "title": chapter["title"],
                    "pov": chapter["pov"],
                    "start_char": start,
                    "end_char": end,
                    "text": chunk,
                    "n_chars": len(chunk),
                }
            )
    return pd.DataFrame(rows)


def load_models(device: str = "cuda") -> LoadedModels:
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )

    embedder = SentenceTransformer(EMBED_MODEL_ID)
    reranker = CrossEncoder(RERANKER_MODEL_ID)

    image_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    image_pipe = image_pipe.to(device)
    image_pipe.enable_attention_slicing()

    return LoadedModels(
        tokenizer=tokenizer,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        image_pipe=image_pipe,
    )


def embed_texts(embedder: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype("float32")


def build_faiss_index(chunks_df: pd.DataFrame, embedder: SentenceTransformer) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    embeddings = embed_texts(embedder, chunks_df["text"].tolist())
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve_passages(
    question: str,
    chunks_df: pd.DataFrame,
    index: faiss.IndexFlatIP,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    top_k: int = 12,
    faiss_k: int = 100,
) -> pd.DataFrame:
    q_emb = embed_texts(embedder, [question], batch_size=1)
    scores, idxs = index.search(q_emb, faiss_k)

    cand = chunks_df.iloc[idxs[0].tolist()].copy()
    cand["faiss_score"] = scores[0]
    pairs = [(question, txt) for txt in cand["text"].tolist()]
    cand["rerank_score"] = reranker.predict(pairs)

    return cand.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)


def build_context(passages_df: pd.DataFrame, max_chars_each: int = 2200) -> str:
    blocks = []
    for _, row in passages_df.iterrows():
        txt = row["text"].strip()[:max_chars_each]
        blocks.append(f"[{row['chunk_id']}] ({row['pov']} | {row['title']})\n{txt}")
    return "\n\n".join(blocks)


def _generate_chat(tokenizer: AutoTokenizer, llm: AutoModelForCausalLM, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    generated = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.03,
    )
    completion_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()


def answer_question(question: str, passages_df: pd.DataFrame, tokenizer: AutoTokenizer, llm: AutoModelForCausalLM) -> str:
    context = build_context(passages_df)
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente experto en el libro 'Juego de Tronos' (Libro 1). "
                "Responde solo con información sustentada en los fragmentos. "
                "Si no está en los fragmentos, di exactamente: 'No encontrado en los fragmentos proporcionados'. "
                "Incluye referencias [chunk_id] al final de cada frase factual."
            ),
        },
        {
            "role": "user",
            "content": f"Pregunta: {question}\n\nFragmentos:\n{context}",
        },
    ]
    return _generate_chat(tokenizer, llm, messages, max_new_tokens=420)


def plan_scene(question: str, answer: str, passages_df: pd.DataFrame, tokenizer: AutoTokenizer, llm: AutoModelForCausalLM) -> dict:
    context = build_context(passages_df, max_chars_each=1400)
    schema = {
        "style": "string",
        "subject": "string",
        "setting": "string",
        "time_of_day": "day|night|dawn|dusk|unknown",
        "mood": "string",
        "characters": [{"name": "string", "appearance": "string", "clothing": "string"}],
        "action": "string",
        "camera": "string",
        "important_objects": ["string"],
        "avoid": ["string"],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an art director. Produce ONLY valid JSON. "
                "Use only details supported by book excerpts + answer. "
                "No actor names, no TV adaptation references."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context}\n\n"
                f"Schema: {json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]
    raw = _generate_chat(tokenizer, llm, messages, max_new_tokens=360)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError(f"No se encontró JSON válido en la salida del planner: {raw[:500]}")
    return json.loads(match.group(0))


def scene_to_prompt(scene: dict) -> tuple[str, str]:
    characters = []
    for c in (scene.get("characters") or [])[:3]:
        desc = ", ".join(filter(None, [c.get("name"), c.get("appearance"), c.get("clothing")]))
        if desc:
            characters.append(desc)

    prompt_parts = [
        "cinematic still, medieval fantasy, high detail, natural lighting",
        scene.get("style", ""),
        f"subject: {scene.get('subject', '')}",
        scene.get("action", ""),
        f"setting: {scene.get('setting', '')}",
        f"time: {scene.get('time_of_day', '')}",
        f"mood: {scene.get('mood', '')}",
        f"characters: {'; '.join(characters)}" if characters else "",
        f"camera: {scene.get('camera', '')}",
        "props: " + ", ".join(scene.get("important_objects", [])) if scene.get("important_objects") else "",
    ]
    prompt = ", ".join([p.strip() for p in prompt_parts if p and str(p).strip()])

    avoid = scene.get("avoid", []) + [
        "text, watermark, logo",
        "tv show actors",
        "modern clothes",
        "low quality, blurry",
    ]
    negative_prompt = ", ".join(dict.fromkeys(avoid))
    return prompt, negative_prompt


def ask_and_draw(question: str, chunks_df: pd.DataFrame, index: faiss.IndexFlatIP, models: LoadedModels, seed: Optional[int] = None):
    passages = retrieve_passages(question, chunks_df, index, models.embedder, models.reranker)
    answer = answer_question(question, passages, models.tokenizer, models.llm)
    scene = plan_scene(question, answer, passages, models.tokenizer, models.llm)
    prompt, negative_prompt = scene_to_prompt(scene)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=models.image_pipe.device).manual_seed(seed)

    image = models.image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=6.0,
        height=1024,
        width=1024,
        generator=generator,
    ).images[0]

    return {
        "answer": answer,
        "passages": passages,
        "scene": scene,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
    }


def save_dataframes(chapters_df: pd.DataFrame, chunks_df: pd.DataFrame, out_dir: str = ".") -> None:
    os.makedirs(out_dir, exist_ok=True)
    chapters_df.to_parquet(os.path.join(out_dir, "chapters.parquet"), index=False)
    chunks_df.to_parquet(os.path.join(out_dir, "chunks.parquet"), index=False)
