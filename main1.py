from fastapi import FastAPI, Request, Header, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import hashlib, hmac, base64, json
import requests
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from datetime import datetime

import os

import re

from qa_engine import index, chunks, query_index, ask_chatgpt
from qa_engine import (
    load_all_docx_from_folder,
    split_into_chunks,
    embed_chunks,
    build_faiss_index,
    query_index
)
import pickle

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://srikaewa.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
# Config
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = Path("faiss_index.pkl")
CHUNKS_PATH = Path("chunks.pkl")

LOG_FILE = "admin.log"
    
# Serve static admin.html at /admin
app.mount("/admin", StaticFiles(directory="static", html=True), name="admin")

# Admin: Upload and process .docx
@app.post("/upload")
async def upload_docx(file: UploadFile = File(...)):
    dest_path = DATA_DIR / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    all_chunks = []
    for docx_file in DATA_DIR.glob("*.docx"):
        paras = load_all_docx_from_folder(str(DATA_DIR))
        all_chunks = split_into_chunks(paras)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    embeddings = embed_chunks(all_chunks)
    index = build_faiss_index(embeddings)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
        
    write_log(f"Uploaded file: {file.filename} and index updated.")
    return {"message": "File uploaded and index updated."}

@app.get("/files")
async def list_files():
    return [f.name for f in DATA_DIR.glob("*.docx")]

def write_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

@app.post("/reindex")
async def reindex_all():
    try:
        all_chunks = []
        for docx_file in DATA_DIR.glob("*.docx"):
            paragraphs = load_all_docx_from_folder(str(DATA_DIR))
            all_chunks = split_into_chunks(paragraphs)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)

        embeddings = embed_chunks(all_chunks)
        index = build_faiss_index(embeddings)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(index, f)

        write_log(f"Reindexed {len(all_chunks)} chunks from {len(list(DATA_DIR.glob('*.docx')))} files.")
        return {"message": "Reindexing complete"}
    except Exception as e:
        write_log(f"❌ Error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reindex")

@app.get("/status")
async def get_status():
    try:
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        last_modified = os.path.getmtime(CHUNKS_PATH)
        updated_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "file_count": len(list(DATA_DIR.glob("*.docx"))),
            "chunk_count": len(chunks),
            "last_updated": updated_str
        }
    except Exception:
        return {
            "file_count": len(list(DATA_DIR.glob("*.docx"))),
            "chunk_count": 0,
            "last_updated": "N/A"
        }

@app.get("/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-10:]
    
@app.delete("/file/{filename}")
async def delete_file(filename: str):
    file_path = DATA_DIR / filename
    if file_path.exists() and file_path.suffix == ".docx":
        file_path.unlink()
        return {"message": f"{filename} deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Web Chat
class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful forensic science assistant."

@app.post("/chat")
async def chat(req: ChatRequest):
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    top_chunks = query_index(index, req.message, chunks)
    context = "\n\n".join(top_chunks)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{req.system}\n{context}"},
            {"role": "user", "content": req.message}
        ]
    )
    return {"response": response.choices[0].message.content.strip()}

@app.get("/health")
async def health():
    return {"status": "ok"}

# LINE webhook with personality and markdown stripping
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# Map @style to personality prompt
style_map = {
    "default": "You are a professional forensic science expert who answers with technical accuracy and clarity.",
    "friendly": "You are a friendly forensic science tutor who explains with warmth and helpful emojis 🧬😊.",
    "kid": "You explain forensic science to a 10-year-old using simple words and fun emojis 🧒🔍🧪.",
    "robotic": "You respond with only facts in a serious, mechanical tone. No emotion or personality 🤖.",
    "casual": "You're a laid-back forensic expert who uses emojis 😎 and humor in your answers."
}

@app.post("/line-webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks, x_line_signature: str = Header(default=None)):
    body = await request.body()

    if not x_line_signature:
        return JSONResponse(status_code=400, content={"error": "Missing LINE signature"})

    hash = hmac.new(LINE_SECRET.encode(), body, hashlib.sha256).digest()
    signature = base64.b64encode(hash).decode()

    if not hmac.compare_digest(signature, x_line_signature):
        return JSONResponse(status_code=403, content={"error": "Invalid signature"})

    data = json.loads(body)
    #print("LINE webhook received:", data)
    background_tasks.add_task(process_line_event, data)
    return {"status": "ok"}

def strip_markdown(md_text):
    clean = re.sub(r'#.*\n', '', md_text)             # remove headings
    clean = re.sub(r'\*\*(.*?)\*\*', r'\\1', clean)  # bold
    clean = re.sub(r'[_*`>]', '', clean)               # other markdown symbols
    clean = re.sub(r'^- ', '', clean, flags=re.MULTILINE)  # bullets
    return clean.strip()

def process_line_event(data):
    try:
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        events = data.get("events", [])
        for event in events:
            if event.get("type") == "message" and event["message"].get("type") == "text":
                raw_text = event["message"]["text"]
                reply_token = event["replyToken"]

                if raw_text.startswith("@"):
                    parts = raw_text.split(" ", 1)
                    style = parts[0][1:].lower()
                    prompt = parts[1] if len(parts) > 1 else ""
                else:
                    style = "default"
                    prompt = raw_text

                system_msg = style_map.get(style, style_map["default"])

                top_chunks = query_index(index, prompt, chunks)
                context = "\n\n".join(top_chunks)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"{system_msg}\n{context}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content.strip()
                plain_answer = strip_markdown(answer)

                res = requests.post(
                    "https://api.line.me/v2/bot/message/reply",
                    headers={
                        "Authorization": f"Bearer {LINE_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "replyToken": reply_token,
                        "messages": [{"type": "text", "text": plain_answer}]
                    }
                )
                print("LINE reply status:", res.status_code, res.text)
    except Exception as e:
        print("❌ Error in background task:", e)

