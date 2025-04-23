from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import os

import markdown


from pydantic import BaseModel
from qa_engine import index, chunks, query_index, ask_chatgpt

FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")

app = FastAPI()

origins = [
    "https://srikaewa.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

#@app.options("/chat")
#async def options_handler():
#    return Response(status_code=204)


@app.post("/chat")
async def chat(req: ChatRequest):
    related_chunks = query_index(index, req.message, chunks)
    context = "\n\n".join(related_chunks)
    answer = ask_chatgpt(context, req.message)
    html_response = markdown.markdown(answer)
    return {"response": html_response}
    
@app.get("/health")
async def health():
    return {"status": "ok"}

#####################
# FB Webhook ########
#####################

@app.get("/fb-webhook")
async def fb_webhook_verify(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == FB_VERIFY_TOKEN:
        return int(challenge)
    return JSONResponse(content={"status": "forbidden"}, status_code=403)

@app.post("/fb-webhook")
async def fb_webhook_handler(request: Request):
    data = await request.json()
    if "entry" in data:
        for entry in data["entry"]:
            for msg_event in entry.get("messaging", []):
                if "message" in msg_event and "text" in msg_event["message"]:
                    sender_id = msg_event["sender"]["id"]
                    user_msg = msg_event["message"]["text"]

                    # RAG logic
                    from qa_engine import index, chunks, query_index, ask_chatgpt
                    top_chunks = query_index(index, user_msg, chunks)
                    context = "\n\n".join(top_chunks)
                    answer = ask_chatgpt(context, user_msg)

                    # Send reply
                    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
                    payload = {
                        "recipient": {"id": sender_id},
                        "message": {"text": answer}
                    }
                    headers = {"Content-Type": "application/json"}
                    requests.post(url, json=payload, headers=headers)
    return {"status": "ok"}