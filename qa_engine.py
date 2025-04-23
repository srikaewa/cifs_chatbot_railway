import os
import openai
import faiss
import numpy as np
from glob import glob
from docx import Document
from dotenv import load_dotenv
from typing import List

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

def load_all_docx_from_folder(folder: str):
    all_paragraphs = []
    for file in glob(os.path.join(folder, "*.docx")):
        doc = Document(file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        all_paragraphs.extend(paragraphs)
    return all_paragraphs

def load_text_from_docx(filepath: str) -> List[str]:
    doc = Document(filepath)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

def split_into_chunks(paragraphs: List[str], max_tokens=500) -> List[str]:
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    chunks.append(current_chunk)
    return chunks

def embed_chunks(chunks: List[str]):
    response = openai.embeddings.create(input=chunks, model=EMBED_MODEL)
    return np.array([r.embedding for r in response.data])

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_index(index, query, chunks, top_k=3):
    query_emb = openai.embeddings.create(input=[query], model=EMBED_MODEL).data[0].embedding
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]

def ask_chatgpt(prompt: str, system_msg: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response['choices'][0]['message']['content'].strip()


# Build once on startup
#paragraphs = load_text_from_docx("data/ความรู้เบื้องต้นด้านนิติวิทยาศาสตร์.docx")
paragraphs = load_all_docx_from_folder("data")
chunks = split_into_chunks(paragraphs)
embeddings = embed_chunks(chunks)
index = build_faiss_index(embeddings)
