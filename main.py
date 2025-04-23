from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

import traceback

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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