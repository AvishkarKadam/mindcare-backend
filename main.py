from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import time

app = FastAPI()

HF_API_KEY = "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"
MODEL = "mistralai/mistral-7b-instruct"

# Basic rate limit per IP (5 messages per 5 minutes)
RATE_LIMIT = 5
RATE_TIME = 300  # seconds

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

    # Safety reminder
    if "suicide" in data.prompt.lower() or "self harm" in data.prompt.lower():
        return {
            "response": "Hey, I’m not a doctor. If you’re in danger, please call 112 right now or talk to someone nearby."
        }

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": data.prompt}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json=payload
    )
    output = response.json()

    if isinstance(output, dict) and "error" in output:
        return {"response": "Model error. Try again later."}

    return {"response": output[0]["generated_text"]}
