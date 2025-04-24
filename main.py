from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import hashlib, hmac, base64
import requests
import os

import traceback

from qa_engine import index, chunks, query_index, ask_chatgpt


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://srikaewa.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful forensic science assistant."

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.message}
            ],
            temperature=0.4
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        print("❌ OpenAI error:", e)
        #traceback.print_exc()
        return {"response": f"⚠️ Server error: {str(e)}"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/line-webhook")
async def line_webhook(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    hash = hmac.new(LINE_SECRET.encode(), body, hashlib.sha256).digest()
    signature = base64.b64encode(hash).decode()

    if not hmac.compare_digest(signature, x_line_signature):
        return JSONResponse(status_code=403, content={"error": "Invalid signature"})

    data = await request.json()
    events = data.get("events", [])

    for event in events:
        if event.get("type") == "message" and event["message"].get("type") == "text":
            raw_text = event["message"]["text"]
            reply_token = event["replyToken"]

            # 🔍 Check for personality prefix like @casual or @kid
            if raw_text.startswith("@"):
                parts = raw_text.split(" ", 1)
                style = parts[0][1:].lower()
                question = parts[1] if len(parts) > 1 else ""
            else:
                style = "default"
                question = raw_text

            system_prompt = style_map.get(style, style_map["default"])
            
            
            #answer = ask_chatgpt(question, system_prompt)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            answer = response.choices[0].message.content.strip()

            # 📤 Send reply to LINE
            requests.post(
                "https://api.line.me/v2/bot/message/reply",
                headers={
                    "Authorization": f"Bearer {LINE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "replyToken": reply_token,
                    "messages": [{"type": "text", "text": answer}]
                }
            )

    return {"status": "ok"}