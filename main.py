from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Depends, Header, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from openai import OpenAI
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pathlib import Path
import os, shutil, pickle, re, json, requests, hmac, hashlib, base64
from typing import List

import base64

from pydantic import BaseModel


from qa_engine import (
    build_chroma_collection, 
    query_collection,
    load_all_docx_from_folder,
    split_into_chunks,
    embed_chunks,
)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Server is starting...")
    yield
    print("🛑 Server is shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://srikaewa.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = "admin.log"

# Mount static admin UI
app.mount("/admin", StaticFiles(directory="static", html=True), name="admin")

# JWT config
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")

# JWT helpers
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username != ADMIN_USERNAME:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Logging
def write_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# Markdown stripping for LINE
def strip_markdown(md_text):
    clean = re.sub(r'#.*\n', '', md_text)
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
    clean = re.sub(r'[_*`>]', '', clean)
    clean = re.sub(r'^- ', '', clean, flags=re.MULTILINE)
    return clean.strip()

@app.on_event("startup")
async def startup_log():
    print("✅ Server is starting...")


# Upload endpoint
@app.post("/upload")
async def upload_docx(file: UploadFile = File(...)):
    dest_path = DATA_DIR / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    write_log(f"Uploaded file: {file.filename}")
    return {"message": "Uploaded"}

# Delete file
@app.delete("/file/{filename}")
async def delete_file(filename: str):
    file_path = DATA_DIR / filename
    if file_path.exists() and file_path.suffix == ".docx":
        file_path.unlink()
        write_log(f"Deleted file: {filename}")
        
        # ✅ Reindex updated knowledge base
        paragraphs = load_all_docx_from_folder(str(DATA_DIR))
        chunks = split_into_chunks(paragraphs)
        build_chroma_collection(chunks)
        write_log("Reindexed after file deletion.")
        
        return {"message": f"{filename} deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

# List uploaded files
@app.get("/files")
async def list_files():
    return [f.name for f in DATA_DIR.glob("*.docx")]

# Reindex route
@app.post("/reindex")
async def reindex_all():
    try:
        paragraphs = load_all_docx_from_folder(str(DATA_DIR))
        chunks = split_into_chunks(paragraphs)
        build_chroma_collection(chunks)
        write_log(f"Reindexed {len(chunks)} chunks from {len(list(DATA_DIR.glob('*.docx')))} files.")
        return {"message": "Reindexing complete"}
    except Exception as e:
        write_log(f"❌ Error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reindex: " + str(e))

@app.get("/status")
async def get_status():
    try:
        num_files = len(list(DATA_DIR.glob("*.docx")))
        from chromadb import Client
        from chromadb.config import Settings
        chroma_client = Client(Settings(persist_directory="./chroma_db"))
        collection = chroma_client.get_collection(name="cifs_collection")
        stats = collection.count()
        return {
            "file_count": num_files,
            "chunk_count": stats,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except:
        return {"file_count": 0, "chunk_count": 0, "last_updated": "N/A"}

@app.get("/logs")
async def get_logs(current_user: str = Depends(get_current_user)):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-10:]

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    #print("🔐 Login attempt:", form_data.username)
    #print("🔐 Provided password:", form_data.password)
    #print("🔐 Stored hash:", ADMIN_PASSWORD_HASH)
    
    #print(f"HASH LEN: {len(ADMIN_PASSWORD_HASH)}")
    #print(f"HASH TEXT: {repr(ADMIN_PASSWORD_HASH)}")

    if form_data.username != ADMIN_USERNAME:
        print("❌ Username mismatch")
        raise HTTPException(status_code=401, detail="Incorrect username")

    if not verify_password(form_data.password, ADMIN_PASSWORD_HASH):
        print("❌ Password mismatch")
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    token = create_access_token(data={"sub": ADMIN_USERNAME})
    print("✅ Login successful")
    return {"access_token": token, "token_type": "bearer"}


class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful forensic science assistant."

@app.post("/chat")
async def chat(req: ChatRequest):
    top_chunks = query_collection(req.message)
    if "No knowledge base" in top_chunks[0]:
        return {"response": "📂 The knowledge base is empty. Please upload and reindex your files first."}
    
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

                top_chunks = query_collection(prompt)
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

