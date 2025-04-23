from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["https://srikaewa.github.io"],
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)

class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful forensic science assistant."

@app.post("/chat")
async def chat(req: ChatRequest):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message}
        ],
        temperature=0.4
    )
    return {"response": response.choices[0].message.content.strip()}

@app.get("/health")
async def health():
    return {"status": "ok"}