# python -m single_agent.memory.router
import os
import time
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from groq import Groq
from dotenv import load_dotenv
import uvicorn
from single_agent.memory.config import llm_max_tokens, llm_temperature

# ============================================================
# 🔧 Load environment
# ============================================================
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in .env")
if not OPENAI_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env (needed for embeddings)")


# ============================================================
# ⚙️ Clients
# ============================================================
groq_client = Groq(api_key=GROQ_KEY)
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


# ============================================================
# 🎨 Colored Logging
# ============================================================
CY = "\033[96m"
GR = "\033[92m"
YL = "\033[93m"
RST = "\033[0m"

def pretty(obj): return json.dumps(obj, indent=2, ensure_ascii=False)

def log_request(tag, body):
    print(f"\n{CY}📥 [{tag}] Request @ {time.strftime('%H:%M:%S')}{RST}")
    print(YL + pretty(body) + RST)

def log_response(tag, body):
    print(f"{GR}📤 [{tag}] Response @ {time.strftime('%H:%M:%S')}{RST}")
    print(GR + pretty(body) + RST)
    print("------------------------------------------------------\n")


# ============================================================
# 🧹 Normalize NON-stream Chat Response
# ============================================================
def normalize_chat_response(completion):
    raw = completion.model_dump()
    choice = raw["choices"][0]
    msg = choice["message"]

    clean_msg = {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content", "")
    }

    if "tool_calls" in msg:
        clean_msg["tool_calls"] = msg["tool_calls"]

    clean_choice = {
        "index": choice.get("index", 0),
        "finish_reason": choice.get("finish_reason", "stop"),
        "message": clean_msg,
    }

    clean = {
        "id": raw.get("id"),
        "object": "chat.completion",
        "created": raw.get("created", int(time.time())),
        "model": raw.get("model"),
        "choices": [clean_choice],
        "usage": raw.get("usage", {})
    }

    return clean


# ============================================================
# 🔄 Stream Normalization (OpenAI SSE format)
# ============================================================
def stream_groq_chat(completion):
    for chunk in completion:
        chunk_dict = chunk.model_dump()

        groq_delta = chunk_dict["choices"][0].get("delta", {})
        delta = {}

        if "content" in groq_delta:
            delta["content"] = groq_delta["content"]
        if "role" in groq_delta:
            delta["role"] = groq_delta["role"]
        if "tool_calls" in groq_delta:
            delta["tool_calls"] = groq_delta["tool_calls"]

        sse = {
            "id": chunk_dict.get("id"),
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta
            }],
        }

        yield f"data: {json.dumps(sse)}\n\n"

    yield "data: [DONE]\n\n"


# ============================================================
# 🤖 Model Map
# ============================================================
MODEL_MAP = {
    "gpt-4o": "openai/gpt-oss-20b",
    "gpt-4o-mini": "openai/gpt-oss-20b",
    "gpt-4o-mini-high": "openai/gpt-oss-20b",
    "gpt-3.5-turbo": "openai/gpt-oss-20b",
}
DEFAULT_MODEL = "openai/gpt-oss-20b"

def map_model(name):
    return MODEL_MAP.get(name, DEFAULT_MODEL)


# ============================================================
# 🚀 FastAPI
# ============================================================
app = FastAPI()


# ============================================================
# 🧠 CHAT COMPLETIONS
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    log_request("Chat", body)

    stream = body.get("stream", False)
    model = map_model(body.get("model"))
    messages = body.get("messages", [])

    # STREAM MODE
    if stream:
        try:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=body.get("temperature", llm_temperature),
                max_tokens=body.get("max_tokens", llm_max_tokens),
                stream=True,
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        return StreamingResponse(stream_groq_chat(completion),
                                 media_type="text/event-stream")

    # NON-STREAM MODE
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=body.get("temperature", llm_temperature),
            max_tokens=body.get("max_tokens", llm_max_tokens),
            stream=False,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    resp = normalize_chat_response(completion)
    log_response("Chat", resp)
    return JSONResponse(resp)


# ============================================================
# ✏️ LEGACY COMPLETIONS
# ============================================================
@app.post("/v1/completions")
async def legacy_completions(request: Request):
    body = await request.json()
    log_request("Completion", body)

    model = map_model(body.get("model"))
    prompt = body.get("prompt", "")

    try:
        completion = groq_client.completions.create(
            model=model,
            prompt=prompt,
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", 2048)
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    resp = {
        "id": completion.id,
        "object": "text_completion",
        "model": model,
        "choices": [{
            "index": 0,
            "text": completion.choices[0].text,
            "finish_reason": completion.choices[0].finish_reason
        }],
        "usage": completion.usage
    }

    log_response("Completion", resp)
    return JSONResponse(resp)


# ============================================================
# 🔍 Embeddings → REAL OPENAI
# ============================================================
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    log_request("Embedding", body)

    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    try:
        response = requests.post(OPENAI_EMBED_URL, json=body, headers=headers)
        resp = response.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    log_response("Embedding", resp)
    return JSONResponse(resp)


# ============================================================
# ▶️ RUN SERVER
# ============================================================
if __name__ == "__main__":
    print(f"{CY}🚀 Hybrid Router Running — Groq LLM + OpenAI Embeddings{RST}")
    uvicorn.run(app, host="0.0.0.0", port=5001)
