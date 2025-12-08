GAIA_TOOL_CANDIDATES = {

    # ============================================================
    # 1) WEB BROWSING (HTTP operations)
    # ============================================================
    "RequestsGetTool": {
        "module": "langchain_community.tools.requests.tool",
        "class": "RequestsGetTool",
        "category": "web-browsing",
        "description": "HTTP GET request tool for web content retrieval."
    },
    "RequestsPostTool": {
        "module": "langchain_community.tools.requests.tool",
        "class": "RequestsPostTool",
        "category": "web-browsing",
        "description": "HTTP POST request tool for submitting web forms or APIs."
    },
    "RequestsPutTool": {
        "module": "langchain_community.tools.requests.tool",
        "class": "RequestsPutTool",
        "category": "web-browsing",
        "description": "HTTP PUT request tool."
    },
    "RequestsPatchTool": {
        "module": "langchain_community.tools.requests.tool",
        "class": "RequestsPatchTool",
        "category": "web-browsing",
        "description": "HTTP PATCH request tool."
    },
    "RequestsDeleteTool": {
        "module": "langchain_community.tools.requests.tool",
        "class": "RequestsDeleteTool",
        "category": "web-browsing",
        "description": "HTTP DELETE request tool."
    },

    # ============================================================
    # 2) SEARCH (no API keys)
    # ============================================================
    "DuckDuckGoSearchRun": {
        "module": "langchain_community.tools.ddg_search.tool",
        "class": "DuckDuckGoSearchRun",
        "category": "search",
        "description": "DuckDuckGo search wrapper; no API key needed."
    },
    "SearxSearchRun": {
        "module": "langchain_community.tools.searx_search.tool",
        "class": "SearxSearchRun",
        "category": "search",
        "description": "Open-source Searx search engine tool."
    },
    "AskNewsSearch": {
        "module": "langchain_community.tools.asknews.tool",
        "class": "AskNewsSearch",
        "category": "search",
        "description": "News search; public endpoints supported."
    },
    "YouSearchTool": {
        "module": "langchain_community.tools.you.tool",
        "class": "YouSearchTool",
        "category": "search",
        "description": "You.com search engine wrapper; no API key needed."
    },
    "BraveSearch": {
        "module": "langchain_community.tools.brave_search.tool",
        "class": "BraveSearch",
        "category": "search",
        "description": "Brave Search wrapper (public endpoint allowed)."
    },

    # ============================================================
    # 3) FILE READING (GAIA requires PDF, CSV, Excel, JSON, XML)
    # ============================================================
    "PyPDFLoader": {
        "module": "langchain_community.document_loaders",
        "class": "PyPDFLoader",
        "category": "file-reading",
        "description": "PDF reader for questions requiring interpreting PDFs."
    },
    "CSVLoader": {
        "module": "langchain_community.document_loaders",
        "class": "CSVLoader",
        "category": "file-reading",
        "description": "CSV file reader."
    },
    "UnstructuredExcelLoader": {
        "module": "langchain_community.document_loaders",
        "class": "UnstructuredExcelLoader",
        "category": "file-reading",
        "description": "Excel spreadsheet reader."
    },
    "TextLoader": {
        "module": "langchain_community.document_loaders",
        "class": "TextLoader",
        "category": "file-reading",
        "description": "Plain text file loader."
    },
    "JSONLoader": {
        "module": "langchain_community.document_loaders",
        "class": "JSONLoader",
        "category": "file-reading",
        "description": "JSON data loader."
    },
    "UnstructuredXMLLoader": {
        "module": "langchain_community.document_loaders",
        "class": "UnstructuredXMLLoader",
        "category": "file-reading",
        "description": "XML/HTML structured loader."
    },

    # ============================================================
    # 4) FILE MANAGEMENT (Dir listing, reading, searching)
    # ============================================================
    "ReadFileTool": {
        "module": "langchain_community.tools.file_management.read",
        "class": "ReadFileTool",
        "category": "file-reading",
        "description": "Read text or binary files."
    },
    "WriteFileTool": {
        "module": "langchain_community.tools.file_management.write",
        "class": "WriteFileTool",
        "category": "file-manipulation",
        "description": "Write data to files."
    },
    "ListDirectoryTool": {
        "module": "langchain_community.tools.file_management.list_dir",
        "class": "ListDirectoryTool",
        "category": "file-manipulation",
        "description": "List directory contents."
    },
    "FileSearchTool": {
        "module": "langchain_community.tools.file_management.file_search",
        "class": "FileSearchTool",
        "category": "file-manipulation",
        "description": "Search for files by pattern."
    },

    # ============================================================
    # 5) CODING / EXECUTION (Python + Shell)
    # ============================================================
    "PythonREPLTool": {
        "module": "langchain_experimental.tools",
        "class": "PythonREPLTool",
        "category": "coding",
        "description": "Executes safe Python code in a sandboxed REPL."
    },
    "ShellTool": {
        "module": "langchain_community.tools.shell.tool",
        "class": "ShellTool",
        "category": "coding",
        "description": "Executes shell commands (sandbox recommended)."
    },

    # ============================================================
    # 6) MULTIMODAL (CUSTOM TOOLS THAT WE WILL IMPLEMENT)
    # ============================================================

    # ---- OCR (Images → Text) ----
    "LocalOCRTool": {
        "module": "custom",
        "class": "LocalOCRTool",
        "category": "multi-modal",
        "description": "OCR extractor using PIL + pytesseract (local)."
    },

    # ---- Image Classification ----
    "LocalImageClassifierTool": {
        "module": "custom",
        "class": "LocalImageClassifierTool",
        "category": "multi-modal",
        "description": "Local image classifier (torchvision MobileNet or CLIP)."
    },

    # ---- Audio Transcription ----
    "LocalAudioTranscriptionTool": {
        "module": "custom",
        "class": "LocalAudioTranscriptionTool",
        "category": "multi-modal",
        "description": "Transcribes audio files using Whisper or Faster-Whisper."
    },

    # ---- Video Frame Extraction ----
    "LocalVideoFrameTool": {
        "module": "custom",
        "class": "LocalVideoFrameTool",
        "category": "multi-modal",
        "description": "Extracts frames from video using ffmpeg or OpenCV."
    },

    # ---- Table Operations (GAIA requires table reasoning) ----
    "LocalTableTool": {
        "module": "custom",
        "class": "LocalTableTool",
        "category": "structured-data",
        "description": "Performs table operations (filtering, aggregation, lookup) using pandas."
    },

    # ---- Symbolic Math (GAIA needs algebra) ----
    "LocalSymPyTool": {
        "module": "custom",
        "class": "LocalSymPyTool",
        "category": "math",
        "description": "Symbolic math operations via sympy."
    },

    # ---- Map / Diagram Understanding ----
    "LocalMapTool": {
        "module": "custom",
        "class": "LocalMapTool",
        "category": "multi-modal",
        "description": "Basic map understanding via OCR + simple geometry."
    },
}
