# StableToolBench Server Setup Guide

## Quick Start

This guide shows how to set up and run the StableToolBench server for CPU-only execution with GPT fallback.

---

## Prerequisites

1. **Python dependencies**:
   ```bash
   pip install fastapi uvicorn python-dotenv openai pyyaml requests slowapi
   ```

2. **OpenAI API Key**: Set in `.env` file (see below)

---

## Step 1: Create `.env` File

Create a `.env` file in the **root folder** (`MASBench/`):

```bash
cd /path/to/MASBench  # Root folder of the project
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

**Important**: The `.env` file should be in the **root folder** (`MASBench/`), not in `single_agent/tool_use/` or `server/` directory.

---

## Step 2: Verify `config.yml`

The `config.yml` file is already configured with:
- Model: `gpt-4o-mini` (CPU-friendly, cost-effective)
- API key: Loaded from `.env` file (leave empty in config)
- Cache folder: `./tool_response_cache`
- Port: `8080`

You can verify/edit `config.yml` if needed:

```yaml
# API key is loaded from .env file (OPENAI_API_KEY)
# Leave empty here - will be read from environment variable
api_key: 
api_base: https://api.openai.com/v1
model: gpt-4o-mini
temperature: 0
toolbench_url: http://8.130.32.149:8080/rapidapi
tools_folder: "./tools"
cache_folder: "./tool_response_cache"
is_save: true
port: 8080
log_file: "./server.log"
```

---

## Step 3: Start the Server

```bash
python single_agent/tool_use/StableToolBench/server/main.py
```

You should see:
```
Loaded .env file from: /path/to/MASBench/.env
OpenAI API key loaded successfully
{'api_key': '', 'api_base': 'https://api.openai.com/v1', ...}
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Server is now running at**: `http://localhost:8080/virtual`

---

## Step 4: Test the Server

In a new terminal, run the test script:

```bash
python single_agent/tool_use/tests/test_server.py
```

The test script will:
1. ✅ Check if server is running
2. ✅ Test cache hit scenarios
3. ✅ Test cache miss + GPT fallback
4. ✅ Validate response format

---

## How It Works

### Execution Flow

```
Framework calls tool
        ↓
POST http://localhost:8080/virtual
        ↓
1) Check cache (JSON file)
        ↓
   Cache hit? → Return cached response (CPU, instant)
        ↓
   Cache miss? → Generate response
        ↓
2) Try real API call (optional, may fail)
        ↓
   Success? → Save to cache, return response
        ↓
   Failed? → Continue to step 3
        ↓
3) GPT fallback (gpt-4o-mini)
        ↓
   Generate fake response using OpenAI API
        ↓
   Save to cache for future use
        ↓
   Return response
```

### Cache Behavior

- **First run**: Cache is empty → GPT generates responses → Saves to cache
- **Subsequent runs**: Cache has responses → Returns instantly (no GPT calls)
- **Cache location**: `tool_response_cache/` (in `single_agent/tool_use/`)
- **Cache structure**: `category/tool_name/api_name.json`

### Benefits

✅ **CPU-only**: No GPU required  
✅ **Cost-effective**: Uses `gpt-4o-mini` (cheaper than `gpt-4-turbo`)  
✅ **Fast after warm-up**: Cache hits are instant  
✅ **Reproducible**: Same inputs = same outputs (from cache)  
✅ **No real API calls**: Uses simulation (unless you configure real API)  

---

## Integration with Your Framework

### Set Environment Variable

When running your framework evaluation, set:

```bash
export SERVICE_URL="http://localhost:8080/virtual"
```

Or in Python:
```python
import os
os.environ['SERVICE_URL'] = 'http://localhost:8080/virtual'
```

### In Your Code

The server will automatically be used by StableToolBench's inference pipeline when:
- `use_mirrorapi_cache=True` is set in `QAPipeline`
- `SERVICE_URL` environment variable points to `http://localhost:8080/virtual`

---

## Troubleshooting

### Issue: "No OpenAI key found"

**Solution**:
- Check that `.env` file exists in the **root folder** (`MASBench/`)
- Verify `OPENAI_API_KEY=your-key` is set correctly
- Restart the server after creating/editing `.env`

### Issue: "Server not responding"

**Solution**:
- Check if server is running: `curl http://localhost:8080/virtual`
- Check server logs for errors
- Verify port 8080 is not in use: `lsof -i :8080`

### Issue: "Cache not working"

**Solution**:
- Check cache folder exists: `ls single_agent/tool_use/tool_response_cache/`
- Verify `is_save: true` in `config.yml`
- Check file permissions on cache folder

### Issue: "GPT calls are slow"

**Solution**:
- This is normal for cache misses (first time)
- Subsequent calls with same parameters will be instant (cache hit)
- Consider pre-warming cache with common API calls

---

## Server Logs

Logs are written to `server.log` (configured in `config.yml`).

Each API call is logged with:
- Timestamp
- Request details
- Response details
- Type (cached_real_response, fake_response, real_response, etc.)

---

## Next Steps

1. ✅ Server is running
2. ✅ Test script passes
3. 🔄 Integrate with your framework evaluation
4. 🔄 Run evaluation pipeline
5. 🔄 Monitor cache growth and GPT usage

---

## Notes

- **First run**: Will make GPT API calls (costs money)
- **After warm-up**: All calls use cache (free, instant)
- **Cache persists**: Between server restarts
- **Cache size**: Grows as more API calls are made
- **Safe to delete cache**: Will regenerate on next run

