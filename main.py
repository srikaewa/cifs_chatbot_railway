
from fastapi import FastAPI, Request, UploadFile, File, Depends, Header, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pathlib import Path
import os, shutil, pickle, re, json, requests, hmac, hashlib, base64
from typing import List

from qa_engine import (
    load_all_docx_from_folder,
    split_into_chunks,
    embed_chunks,
    build_faiss_index,
    query_index
)

app = FastAPI()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = Path("faiss_index.pkl")
CHUNKS_PATH = Path("chunks.pkl")
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
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", pwd_context.hash("admin123"))

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
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(index, f)
        write_log(f"Reindexed {len(chunks)} chunks from {len(list(DATA_DIR.glob('*.docx')))} files.")
        return {"message": "Reindexing complete"}
    except Exception as e:
        write_log(f"❌ Error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reindex")

# Status
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
    except:
        return {"file_count": 0, "chunk_count": 0, "last_updated": "N/A"}

# Secure logs
@app.get("/logs")
async def get_logs(current_user: str = Depends(get_current_user)):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-10:]

# Login
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or not verify_password(form_data.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(data={"sub": ADMIN_USERNAME})
    return {"access_token": token, "token_type": "bearer"}
