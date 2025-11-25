# python -m single_agent.memory.router_local

import os
import time
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn

from single_agent.memory.config import (
    llm_max_tokens,
    llm_temperature,
    KEEP_ANALYSIS,
)

# ============================================================
# 🔧 Load environment
# ============================================================
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # ONLY for embeddings

if not OPENAI_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env (needed for embeddings)")

# ============================================================
# 🌐 Llama.cpp server config
# ============================================================
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://206.12.92.147:22101")
LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "/shared_mnt/OpenAi/gpt-oss-20b-Q5_K_M.gguf",
)

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

# ============================================================
# 🎨 Colored Logging
# ============================================================
CY = "\033[96m"
GR = "\033[92m"
YL = "\033[93m"
RST = "\033[0m"

def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

# ============================================================
# 🔧 Analysis filtering
# ============================================================
def maybe_strip_analysis(messages):
    if KEEP_ANALYSIS:
        return messages

    cleaned = []
    for m in messages:
        if m["role"] == "system":
            text = m["content"]
            text = text.replace(
                "Respond directly in the <|assistant|>final channel only. Do not output analysis or chain of thought.",
                ""
            )
            text = text.replace("<|analysis|>", "")
            text = text.replace("<|assistant|>analysis", "")
            cleaned.append({"role": m["role"], "content": text.strip()})
        else:
            cleaned.append(m)
    return cleaned


def filter_analysis_from_content(content):
    if not content or KEEP_ANALYSIS:
        return content

    text = str(content)

    if "<|analysis|>" in text and "<|final|>" in text:
        return text.split("<|final|>")[-1].strip()

    if "<|analysis|>" in text:
        before, after = text.split("<|analysis|>", 1)
        return before.strip() or after.strip()

    if "<|thinking|>" in text:
        before, after = text.split("<|thinking|>", 1)
        return before.strip() or after.strip()

    return text


# ============================================================
# 📝 Logging
# ============================================================
def log_request(tag, body):
    try:
        log_body = json.loads(json.dumps(body))
    except Exception:
        log_body = body

    print(f"\n{CY}📥 [{tag}] Request @ {time.strftime('%H:%M:%S')}{RST}")
    print(YL + pretty(log_body) + RST)


def log_response(tag, body):
    try:
        log_body = json.loads(json.dumps(body))
    except Exception:
        log_body = body

    print(f"{GR}📤 [{tag}] Response @ {time.strftime('%H:%M:%S')}{RST}")
    print(GR + pretty(log_body) + RST)
    print("------------------------------------------------------\n")


# ============================================================
# 🚀 Model mapping
# ============================================================
MODEL_MAP = {
    "gpt-4o": LLAMA_MODEL,
    "gpt-4o-mini": LLAMA_MODEL,
    "gpt-4o-mini-high": LLAMA_MODEL,
    "gpt-3.5-turbo": LLAMA_MODEL,
}
DEFAULT_MODEL = LLAMA_MODEL

def map_model(name: str | None) -> str:
    return MODEL_MAP.get(name, DEFAULT_MODEL)


def clamp_temperature(t):
    try:
        return max(0.0, min(2.0, float(t)))
    except Exception:
        return llm_temperature


# ============================================================
# 🚀 FastAPI App
# ============================================================
app = FastAPI()


# ============================================================
# 🧠 NON-STREAMING /v1/chat/completions
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    log_request("Chat", body)

    # 🚫 force NON-STREAM ALWAYS
    body["stream"] = False

    model = map_model(body.get("model"))
    messages = body.get("messages", [])

    messages = maybe_strip_analysis(messages)

    temperature = clamp_temperature(body.get("temperature", llm_temperature))
    max_tokens = body.get("max_tokens", llm_max_tokens)

    body["model"] = model
    body["messages"] = messages
    body["temperature"] = temperature
    body["max_tokens"] = max_tokens
    body["stop"] = ["<|end_of_text|>"]

    try:
        llama_resp = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json=body,
            timeout=600,
        )
        resp_json = llama_resp.json()

        # 🔥 Strip analysis in completion
        if not KEEP_ANALYSIS and "choices" in resp_json:
            for choice in resp_json["choices"]:
                msg = choice.get("message", {})
                if "content" in msg:
                    msg["content"] = filter_analysis_from_content(msg["content"])

    except Exception as e:
        err = {"error": str(e)}
        log_response("Chat-Error", err)
        return JSONResponse(err, status_code=500)

    log_response("Chat", resp_json)
    return JSONResponse(resp_json)



# ============================================================
# ✏️ NON-STREAM /v1/completions
# ============================================================
@app.post("/v1/completions")
async def legacy_completions(request: Request):
    body = await request.json()
    log_request("Completion", body)

    body["model"] = map_model(body.get("model"))

    try:
        llama_resp = requests.post(
            f"{LLAMA_SERVER_URL}/v1/completions",
            json=body,
            timeout=600,
        )
        resp = llama_resp.json()

    except Exception as e:
        err = {"error": str(e)}
        log_response("Completion-Error", err)
        return JSONResponse(err, status_code=500)

    log_response("Completion", resp)
    return JSONResponse(resp)



# ============================================================
# 🔍 OpenAI Embeddings (real)
# ============================================================
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    log_request("Embedding", body)

    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}

    try:
        response = requests.post(
            OPENAI_EMBED_URL,
            json=body,
            headers=headers
        )
        resp = response.json()

    except Exception as e:
        err = {"error": str(e)}
        log_response("Embedding-Error", err)
        return JSONResponse(err, status_code=500)

    log_response("Embedding", resp)
    return JSONResponse(resp)



# ============================================================
# ▶️ Run server
# ============================================================
if __name__ == "__main__":
    print(f"{CY}🚀 GPT-OSS Router Running (NO STREAMING){RST}")
    print(f"{CY}🌐 Llama server: {LLAMA_SERVER_URL}, model: {LLAMA_MODEL}{RST}")
    print(f"{CY}📊 Analysis filtering: {'KEEPING' if KEEP_ANALYSIS else 'STRIPPING'}{RST}")

    uvicorn.run(app, host="0.0.0.0", port=5001)
