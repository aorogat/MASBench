"""
Extra custom tools for GAIA-style multimodal capabilities.

These tools are intentionally framework-agnostic: each is a small
Python class with a __call__ method. They can be wrapped by LangChain,
CrewAI, LangGraph, or OpenAI Agents as needed.

All tools are "best-effort": if an optional dependency is missing,
they return a clear error message instead of crashing.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


# -------------------------------------------------------------------
# Helper: generic result wrapper (optional, can just return str as well)
# -------------------------------------------------------------------


@dataclass
class ToolResult:
    """Simple result container for custom tools."""
    success: bool
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        prefix = "[OK]" if self.success else "[ERROR]"
        return f"{prefix} {self.content}"


# -------------------------------------------------------------------
# LocalOCRTool  (PIL + pytesseract)
# -------------------------------------------------------------------


class LocalOCRTool:
    """
    Local OCR tool using PIL + pytesseract.

    Usage:
        tool = LocalOCRTool()
        result = tool("path/to/image.png")
        print(result)
    """

    name = "LocalOCRTool"
    description = "Extracts text from images using local Tesseract OCR."

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd

    def __call__(self, image_path: str) -> ToolResult:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                content=(
                    "LocalOCRTool requires 'pillow' and 'pytesseract'. "
                    "Install them with: pip install pillow pytesseract"
                ),
            )

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        if not os.path.exists(image_path):
            return ToolResult(False, f"Image file not found: {image_path}")

        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return ToolResult(True, text.strip(), {"path": image_path})
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(False, f"OCR failed: {e!r}")


# -------------------------------------------------------------------
# LocalImageClassifierTool  (torchvision)
# -------------------------------------------------------------------


class LocalImageClassifierTool:
    """
    Very lightweight image classification tool using torchvision.

    It loads a pretrained MobileNetV2 (or similar) on first use.
    This is only meant for small-scale experiments, not production.

    Usage:
        tool = LocalImageClassifierTool()
        result = tool("image.jpg")
        print(result.content)
    """

    name = "LocalImageClassifierTool"
    description = "Classifies images using a local torchvision model."

    def __init__(self, top_k: int = 3, device: Optional[str] = None):
        self.top_k = top_k
        self.device = device or "cpu"
        self._model = None
        self._transform = None
        self._labels = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from torchvision import models, transforms  # type: ignore
        except ImportError:
            raise RuntimeError(
                "LocalImageClassifierTool requires 'torch' and 'torchvision'. "
                "Install them with: pip install torch torchvision --extra-index-url "
                "https://download.pytorch.org/whl/cpu"
            )

        # lazy-load model and labels
        self._model = models.mobilenet_v2(pretrained=True)
        self._model.eval()
        self._model.to(self.device)

        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Minimal ImageNet labels list; you can replace with full mapping if needed.
        self._labels = [f"class_{i}" for i in range(1000)]

    def __call__(self, image_path: str) -> ToolResult:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return ToolResult(
                False,
                "LocalImageClassifierTool requires 'pillow'. "
                "Install with: pip install pillow",
            )

        if not os.path.exists(image_path):
            return ToolResult(False, f"Image file not found: {image_path}")

        try:
            self._lazy_init()
        except RuntimeError as e:
            return ToolResult(False, str(e))

        import torch  # type: ignore

        try:
            img = Image.open(image_path).convert("RGB")
            x = self._transform(img).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                logits = self._model(x)  # type: ignore
                probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, k=self.top_k)
            indices = topk.indices.cpu().tolist()
            values = topk.values.cpu().tolist()
            labels_probs = [
                (self._labels[i], float(p)) for i, p in zip(indices, values)
            ]
            text = ", ".join(
                f"{label} (p={p:.3f})" for label, p in labels_probs
            )
            return ToolResult(
                True,
                text,
                {"path": image_path, "predictions": labels_probs},
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(False, f"Image classification failed: {e!r}")


# -------------------------------------------------------------------
# LocalAudioTranscriptionTool  (Whisper / Faster-Whisper)
# -------------------------------------------------------------------


class LocalAudioTranscriptionTool:
    """
    Transcribes audio files using whisper / faster-whisper if installed.

    Usage:
        tool = LocalAudioTranscriptionTool()
        result = tool("clip.m4a")
    """

    name = "LocalAudioTranscriptionTool"
    description = "Transcribes audio locally using Whisper / Faster-Whisper."

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _lazy_init_whisper(self):
        # Try faster-whisper first, then openai-whisper
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = ("faster-whisper", WhisperModel(self.model_size, device="cpu"))
            return
        except ImportError:
            pass

        try:
            import whisper  # type: ignore

            self._model = ("whisper", whisper.load_model(self.model_size))
            return
        except ImportError:
            raise RuntimeError(
                "LocalAudioTranscriptionTool requires 'faster-whisper' or 'whisper'. "
                "Install with: pip install faster-whisper  OR  pip install openai-whisper"
            )

    def __call__(self, audio_path: str) -> ToolResult:
        if not os.path.exists(audio_path):
            return ToolResult(False, f"Audio file not found: {audio_path}")

        try:
            self._lazy_init_whisper()
        except RuntimeError as e:
            return ToolResult(False, str(e))

        backend, model = self._model  # type: ignore

        try:
            if backend == "faster-whisper":
                segments, info = model.transcribe(audio_path)
                text = " ".join(seg.text.strip() for seg in segments)
                return ToolResult(
                    True, text.strip(), {"duration": info.duration, "language": info.language}
                )
            else:
                # openai-whisper
                result = model.transcribe(audio_path)
                return ToolResult(
                    True,
                    result.get("text", "").strip(),
                    {"language": result.get("language")},
                )
        except Exception as e:  # pragma: no cover
            return ToolResult(False, f"Transcription failed: {e!r}")


# -------------------------------------------------------------------
# LocalVideoFrameTool  (ffmpeg / OpenCV)
# -------------------------------------------------------------------


class LocalVideoFrameTool:
    """
    Extracts a frame from a video file.

    By default, extracts a frame at a given timestamp (in seconds)
    and optionally saves it as an image.

    Usage:
        tool = LocalVideoFrameTool()
        result = tool("video.mp4", timestamp=3.0)
    """

    name = "LocalVideoFrameTool"
    description = "Extracts frames from video using OpenCV (or imageio fallback)."

    def __call__(
        self,
        video_path: str,
        timestamp: float = 0.0,
        output_image_path: Optional[str] = None,
    ) -> ToolResult:
        if not os.path.exists(video_path):
            return ToolResult(False, f"Video file not found: {video_path}")

        # Try OpenCV first
        try:
            import cv2  # type: ignore
        except ImportError:
            cv2 = None  # type: ignore

        try:
            if cv2 is not None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return ToolResult(False, "Could not open video with OpenCV.")

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_idx = int(max(0, math.floor(timestamp * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                cap.release()

                if not ok or frame is None:
                    return ToolResult(False, "Could not read frame at requested timestamp.")

                if output_image_path:
                    cv2.imwrite(output_image_path, frame)
                    meta = {"frame_index": frame_idx, "saved_to": output_image_path}
                    return ToolResult(True, f"Frame saved to {output_image_path}", meta)
                else:
                    meta = {"frame_index": frame_idx, "shape": frame.shape}
                    return ToolResult(True, "Frame extracted successfully.", meta)

            # Fallback: imageio
            import imageio  # type: ignore

            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data().get("fps", 25.0)
            frame_idx = int(max(0, math.floor(timestamp * fps)))
            frame = reader.get_data(frame_idx)
            reader.close()

            if output_image_path:
                imageio.imwrite(output_image_path, frame)
                meta = {"frame_index": frame_idx, "saved_to": output_image_path}
                return ToolResult(True, f"Frame saved to {output_image_path}", meta)
            else:
                meta = {"frame_index": frame_idx, "shape": frame.shape}
                return ToolResult(True, "Frame extracted successfully.", meta)

        except ImportError:
            return ToolResult(
                False,
                "LocalVideoFrameTool requires 'opencv-python' or 'imageio'. "
                "Install with: pip install opencv-python imageio",
            )
        except Exception as e:  # pragma: no cover
            return ToolResult(False, f"Video frame extraction failed: {e!r}")


# -------------------------------------------------------------------
# LocalTableTool  (pandas)
# -------------------------------------------------------------------


class LocalTableTool:
    """
    Performs basic table operations on CSV / Excel using pandas.

    Supported operations:
        - 'head': first N rows
        - 'describe': numeric summary
        - 'column': select one column
        - 'value_counts': frequency counts of a column

    Usage:
        tool = LocalTableTool()
        result = tool("file.csv", op="head", n=5)
    """

    name = "LocalTableTool"
    description = "Runs simple table operations (head, describe, value_counts) using pandas."

    def __call__(
        self,
        table_path: str,
        op: str = "head",
        n: int = 5,
        column: Optional[str] = None,
    ) -> ToolResult:
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return ToolResult(
                False,
                "LocalTableTool requires 'pandas'. Install with: pip install pandas",
            )

        if not os.path.exists(table_path):
            return ToolResult(False, f"Table file not found: {table_path}")

        ext = os.path.splitext(table_path)[1].lower()
        try:
            if ext in [".csv", ".tsv"]:
                df = pd.read_csv(table_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(table_path)
            else:
                return ToolResult(False, f"Unsupported table extension: {ext}")
        except Exception as e:
            return ToolResult(False, f"Failed to load table: {e!r}")

        try:
            if op == "head":
                return ToolResult(True, df.head(n).to_string())
            if op == "describe":
                return ToolResult(True, df.describe(include="all").to_string())
            if op == "column":
                if column is None or column not in df.columns:
                    return ToolResult(False, f"Column '{column}' not found.")
                return ToolResult(True, df[column].to_string())
            if op == "value_counts":
                if column is None or column not in df.columns:
                    return ToolResult(False, f"Column '{column}' not found.")
                vc = df[column].value_counts()
                return ToolResult(True, vc.to_string())
            return ToolResult(False, f"Unsupported operation: {op}")
        except Exception as e:  # pragma: no cover
            return ToolResult(False, f"Table operation failed: {e!r}")


# -------------------------------------------------------------------
# LocalSymPyTool  (sympy)
# -------------------------------------------------------------------


class LocalSymPyTool:
    """
    Symbolic math helper using sympy.

    Supports:
        - simplify
        - solve (for x by default)

    Usage:
        tool = LocalSymPyTool()
        result = tool("x^2 - 4", op="solve")
    """

    name = "LocalSymPyTool"
    description = "Symbolic math operations via sympy (simplify, solve)."

    def __call__(
        self,
        expression: str,
        op: str = "simplify",
        variable: str = "x",
    ) -> ToolResult:
        try:
            import sympy as sp  # type: ignore
        except ImportError:
            return ToolResult(
                False,
                "LocalSymPyTool requires 'sympy'. Install with: pip install sympy",
            )

        try:
            x = sp.symbols(variable)
            expr = sp.sympify(expression)
        except Exception as e:
            return ToolResult(False, f"Could not parse expression: {e!r}")

        try:
            if op == "simplify":
                out = sp.simplify(expr)
                return ToolResult(True, str(out))
            if op == "solve":
                sol = sp.solve(sp.Eq(expr, 0), x)
                return ToolResult(True, str(sol))
            return ToolResult(False, f"Unsupported operation: {op}")
        except Exception as e:  # pragma: no cover
            return ToolResult(False, f"SymPy operation failed: {e!r}")


# -------------------------------------------------------------------
# LocalMapTool  (very simple: OCR + hints)
# -------------------------------------------------------------------


class LocalMapTool:
    """
    Basic map/diagram understanding.

    For now this is a wrapper around LocalOCRTool that simply extracts
    any text from the map image. Higher-level reasoning is delegated
    to the LLM using the extracted text.

    Usage:
        tool = LocalMapTool()
        result = tool("map.png")
    """

    name = "LocalMapTool"
    description = "Very lightweight map helper built on top of OCR."

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.ocr = LocalOCRTool(tesseract_cmd=tesseract_cmd)

    def __call__(self, image_path: str) -> ToolResult:
        ocr_result = self.ocr(image_path)
        if not ocr_result.success:
            return ToolResult(False, f"Map OCR failed: {ocr_result.content}")
        # In future, you could add heuristics for directions, distances, etc.
        return ToolResult(
            True,
            ocr_result.content,
            {"source": "LocalMapTool", "path": image_path},
        )


# -------------------------------------------------------------------
# Minimal self-test when run as a script
# -------------------------------------------------------------------


def _self_test() -> None:
    """Quick smoke-test to verify imports and basic behavior."""
    print("Running extra_tools self-test...\n")

    # We don't assume any real media files exist; we just check error paths.
    ocr = LocalOCRTool()
    print("LocalOCRTool:", ocr("nonexistent.png"))

    img_cls = LocalImageClassifierTool(top_k=1)
    print("LocalImageClassifierTool:", img_cls("nonexistent.png"))

    audio = LocalAudioTranscriptionTool()
    print("LocalAudioTranscriptionTool:", audio("nonexistent.m4a"))

    video = LocalVideoFrameTool()
    print("LocalVideoFrameTool:", video("nonexistent.mp4"))

    table = LocalTableTool()
    print("LocalTableTool:", table("nonexistent.csv", op="head"))

    sym = LocalSymPyTool()
    print("LocalSymPyTool (simplify):", sym("x^2 - 1", op="simplify"))
    print("LocalSymPyTool (solve):", sym("x^2 - 1", op="solve"))

    mtool = LocalMapTool()
    print("LocalMapTool:", mtool("nonexistent.png"))


if __name__ == "__main__":
    _self_test()
