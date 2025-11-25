# python -m single_agent.memory.router_groq
import os
import time
import json
import asyncio
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from groq import Groq
from dotenv import load_dotenv
import uvicorn

from single_agent.memory.config import (
    llm_max_tokens,
    llm_temperature,
)

# ============================================================
# 🔧 Load environment
# ============================================================
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # used ONLY for embeddings

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


def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)


def log_request(tag, body):
    # Make a deep copy ONLY for printing
    try:
        log_body = json.loads(json.dumps(body))  # safe deep copy
    except Exception:
        log_body = body  # fallback: print raw

    # Special pretty-print compression for embeddings ONLY in the log output
    if tag == "Embedding" and isinstance(log_body, dict):
        if "input" in log_body:

            inp = log_body["input"]

            def short_text(x):
                if not isinstance(x, str):
                    return x
                x = x.replace("\n", " ").replace("\r", " ").strip()
                return (x[:80] + "...") if len(x) > 80 else x

            # Case 1: input is a string
            if isinstance(inp, str):
                log_body["input"] = short_text(inp)

            # Case 2: input is a list
            elif isinstance(inp, list):
                short_list = []
                for item in inp[:2]:     # only first 2 items
                    if isinstance(item, str):
                        short_list.append(short_text(item))
                    elif isinstance(item, list):
                        short_list.append(f"[{len(item)} tokens]")
                    else:
                        short_list.append(item)
                if len(inp) > 2:
                    short_list.append("...")  # indicate truncation
                log_body["input"] = short_list

    print(f"\n{CY}📥 [{tag}] Request @ {time.strftime('%H:%M:%S')}{RST}")
    print(YL + pretty(log_body) + RST)



def log_response(tag, body):
    # Deep copy only for printing
    try:
        log_body = json.loads(json.dumps(body))
    except Exception:
        log_body = body

    # Truncate embedding vectors ONLY in print
    if isinstance(log_body, dict) and "data" in log_body:
        for item in log_body.get("data", []):
            if isinstance(item, dict) and "embedding" in item:
                emb = item["embedding"]

                # Numeric vector
                if isinstance(emb, list):
                    item["embedding"] = f"[vector len={len(emb)}, first 3={emb[:3]} ...]"

                # Base64 string
                elif isinstance(emb, str):
                    item["embedding"] = emb[:40] + "..."

    print(f"{GR}📤 [{tag}] Response @ {time.strftime('%H:%M:%S')}{RST}")
    print(GR + pretty(log_body) + RST)
    print("------------------------------------------------------\n")


# ============================================================
# 🧹 Normalize NON-stream Chat Response (OpenAI style)
# ============================================================
def normalize_chat_response(completion):
    """
    Convert Groq ChatCompletion object → OpenAI-compatible /v1/chat/completions JSON.
    """
    raw = completion.model_dump()
    choice = raw["choices"][0]
    msg = choice["message"]

    clean_msg = {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content", ""),
    }

    # Preserve tool_calls if model/tooling is used
    if "tool_calls" in msg and msg["tool_calls"] is not None:
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
        "usage": raw.get("usage", {}),
    }

    return clean


# ============================================================
# 🔄 Streaming normalization → OpenAI SSE format
# ============================================================
# ============================================================
# 🧹 Helper to clean Groq internal fields from logs
# ============================================================
def clean_for_log(chunk):
    """
    Remove Groq internal fields from logged chunk only.
    This ensures logs show exactly what agents receive.
    """
    if "choices" in chunk:
        for c in chunk["choices"]:
            delta = c.get("delta", {})
            if isinstance(delta, dict):
                # Remove all Groq-specific fields
                delta.pop("reasoning", None)
                delta.pop("channel", None)
                delta.pop("annotations", None)
                delta.pop("executed_tools", None)
                delta.pop("analysis", None)
    
    # Remove top-level Groq metadata
    chunk.pop("x_groq", None)
    return chunk


# ============================================================
# 🔄 Fixed Streaming normalization → OpenAI SSE format
# ============================================================
async def sse_event_stream(completion):
    """
    Convert Groq's streaming response → OpenAI-compatible SSE events.
    - Filters out Groq reasoning-only chunks (no content)
    - Logs clean, OpenAI-style chunks
    - Ensures agents always receive valid assistant messages
    - Sends fallback message if Groq returns no content
    """
    first_chunk_sent = False
    has_sent_content = False  # Track if any real content was sent
    print(f"{GR}📤 [Chat-Stream] START @ {time.strftime('%H:%M:%S')}{RST}")

    try:
        for chunk in completion:
            # --- Get raw chunk data ---
            chunk_dict = chunk.model_dump()
            groq_choice = chunk_dict["choices"][0]
            groq_delta = groq_choice.get("delta", {}) or {}

            # ✅ FIX #1: Skip Groq reasoning-only chunks (no content)
            # This is the root cause of "No assistant response collected"
            # Filter out both None AND empty strings
            if "content" not in groq_delta or not groq_delta["content"]:
                continue

            # ✅ FIX #2: Ensure first valid chunk has role
            if not first_chunk_sent:
                if "role" not in groq_delta:
                    groq_delta["role"] = "assistant"
                first_chunk_sent = True

            # --- Build OpenAI-compatible delta ---
            delta = {}

            if "role" in groq_delta:
                delta["role"] = groq_delta["role"]

            # Content is guaranteed to exist due to filter above
            delta["content"] = groq_delta["content"]

            # Preserve tool_calls if present
            if "tool_calls" in groq_delta and groq_delta["tool_calls"] is not None:
                delta["tool_calls"] = groq_delta["tool_calls"]

            # --- Build OpenAI-compatible choice ---
            choice_payload = {
                "index": groq_choice.get("index", 0),
                "delta": delta,
                "finish_reason": groq_choice.get("finish_reason"),
            }

            # --- Build OpenAI-compatible SSE event ---
            sse = {
                "id": chunk_dict.get("id"),
                "object": "chat.completion.chunk",
                "created": chunk_dict.get("created", int(time.time())),
                "model": chunk_dict.get("model"),
                "choices": [choice_payload],
            }

            # ✅ FIX #3: Log the cleaned chunk (matches what agent receives)
            try:
                log_chunk = clean_for_log(json.loads(json.dumps(sse)))
                
                # Truncate long content in log for readability
                log_delta = log_chunk["choices"][0].get("delta", {})
                if "content" in log_delta and isinstance(log_delta["content"], str):
                    if len(log_delta["content"]) > 100:
                        log_delta["content"] = log_delta["content"][:100] + "..."
                
                print(GR + pretty(log_chunk) + RST)
            except Exception:
                pass

            # --- Emit SSE event ---
            yield f"data: {json.dumps(sse)}\n\n"
            has_sent_content = True  # Mark that we sent real content
            await asyncio.sleep(0)

        # ✅ FIX #4: If Groq sent no content, send a fallback message
        if not has_sent_content:
            print(f"{YL}⚠️  Groq returned no content - sending fallback message{RST}")
            
            fallback_sse = {
                "id": f"fallback-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "openai/gpt-oss-20b",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "I apologize, but I'm unable to generate a response for this query. Could you rephrase or provide more context?"
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(fallback_sse)}\n\n"
            
            # Send finish chunk
            finish_sse = {
                "id": f"fallback-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "openai/gpt-oss-20b",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(finish_sse)}\n\n"

        print(f"{GR}📤 [Chat-Stream] END @ {time.strftime('%H:%M:%S')}{RST}")
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"{GR}📤 [Chat-Stream] ERROR: {e}{RST}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
# ============================================================
# 🤖 Model Map (OpenAI → Groq)
# ============================================================
MODEL_MAP = {
    "gpt-4o": "openai/gpt-oss-20b",
    "gpt-4o-mini": "openai/gpt-oss-20b",
    "gpt-4o-mini-high": "openai/gpt-oss-20b",
    "gpt-3.5-turbo": "openai/gpt-oss-20b",
}
DEFAULT_MODEL = "openai/gpt-oss-20b"


def map_model(name: str | None) -> str:
    if not name:
        return DEFAULT_MODEL
    return MODEL_MAP.get(name, DEFAULT_MODEL)


def clamp_temperature(t):
    try:
        t = float(t)
    except Exception:
        t = llm_temperature
    # Groq requires 0–2
    return max(0.0, min(2.0, t))



# ============================================================
# 🚀 FastAPI app
# ============================================================
app = FastAPI()


# ============================================================
# 🧠 /v1/chat/completions → Groq
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    log_request("Chat", body)

    stream = body.get("stream", False)
    model = map_model(body.get("model"))
    messages = body.get("messages", [])



    temperature = clamp_temperature(body.get("temperature", llm_temperature))
    max_tokens = body.get("max_tokens", llm_max_tokens)

    # STREAM MODE (used heavily by LangGraph)
    if stream:
        try:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as e:
            err = {"error": str(e)}
            log_response("Chat-Error", err)
            return JSONResponse(err, status_code=500)

        return StreamingResponse(
            sse_event_stream(completion),
            media_type="text/event-stream",
        )

    # NON-STREAM MODE (CrewAI, Agno, etc.)
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as e:
        err = {"error": str(e)}
        log_response("Chat-Error", err)
        return JSONResponse(err, status_code=500)

    resp = normalize_chat_response(completion)
    log_response("Chat", resp)
    return JSONResponse(resp)


# ============================================================
# ✏️ /v1/completions → Groq (legacy text completions)
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
            temperature=clamp_temperature(body.get("temperature", llm_temperature)),
            max_tokens=body.get("max_tokens", llm_max_tokens),
        )
    except Exception as e:
        err = {"error": str(e)}
        log_response("Completion-Error", err)
        return JSONResponse(err, status_code=500)

    resp = {
        "id": completion.id,
        "object": "text_completion",
        "model": model,
        "choices": [{
            "index": 0,
            "text": completion.choices[0].text,
            "finish_reason": completion.choices[0].finish_reason,
        }],
        "usage": completion.usage,
    }

    log_response("Completion", resp)
    return JSONResponse(resp)


# ============================================================
# 🔍 /v1/embeddings → Real OpenAI
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
        err = {"error": str(e)}
        log_response("Embedding-Error", err)
        return JSONResponse(err, status_code=500)

    log_response("Embedding", resp)
    return JSONResponse(resp)


# ============================================================
# ▶️ Run server
# ============================================================
if __name__ == "__main__":
    print(f"{CY}🚀 Hybrid Router Running — Groq LLM + OpenAI Embeddings{RST}")
    uvicorn.run(app, host="0.0.0.0", port=5001)
