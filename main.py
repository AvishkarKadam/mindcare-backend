from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "google/flan-t5-small"

RATE_LIMIT = 5
RATE_TIME = 300
ip_logs = {}

class Chat(BaseModel):
    prompt: str

def check_rate(ip):
    now = time.time()
    if ip not in ip_logs:
        ip_logs[ip] = []
    ip_logs[ip] = [t for t in ip_logs[ip] if now - t < RATE_TIME]
    if len(ip_logs[ip]) >= RATE_LIMIT:
        return False
    ip_logs[ip].append(now)
    return True

@app.post("/chat")
async def chat(data: Chat, request: Request):
    ip = request.client.host
    if not check_rate(ip):
        return {"error": "Too many requests. Please wait a bit."}

    # Safety
    if "suicide" in data.prompt.lower() or "self harm" in data.prompt.lower():
        return {
            "response": "Hey, I’m not a doctor. If you’re in danger, please call 112 right now or talk to someone nearby."
        }

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # System prompt for emotional support
    system_prompt = (
        "You are a friendly, emotional, supportive AI. "
        "Talk like a caring friend. "
        "Make long, comforting responses. "
        "Ask follow-up questions gently. "
        "Avoid giving medical advice. "
        "If user is in danger, tell them to seek help immediately."
    )

    response = requests.post(
        "https://router.huggingface.co/api/v1/runs",
        headers=headers,
        json={
            "model": MODEL,
            "inputs": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.prompt}
            ],
            "parameters": {"max_new_tokens": 250}
        }
    )

    output = response.json()

    if "error" in output:
        return {"response": "Model error. Try again later."}

    return {"response": output["generated_text"]}

