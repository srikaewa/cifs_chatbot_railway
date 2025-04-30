
import os
import openai
import chromadb
import numpy as np
from glob import glob
from docx import Document
from dotenv import load_dotenv
from typing import List
from chromadb.config import Settings

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))
COLLECTION_NAME = "cifs_collection"

def load_all_docx_from_folder(folder: str):
    all_paragraphs = []
    for file in glob(os.path.join(folder, "*.docx")):
        doc = Document(file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        all_paragraphs.extend(paragraphs)
    return all_paragraphs

def split_into_chunks(paragraphs: List[str], max_tokens=500):
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

def build_chroma_collection(chunks: List[str]):
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    embeddings = embed_chunks(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids,
    )
    chroma_client.persist()
    return collection

def query_collection(query: str, top_k=3):
    if COLLECTION_NAME not in [c.name for c in chroma_client.list_collections()]:
        chroma_client.create_collection(name=COLLECTION_NAME)
        return ["No knowledge base available. Please upload and reindex."]

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    query_emb = embed_chunks([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents"]
    )
    return results["documents"][0]

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
