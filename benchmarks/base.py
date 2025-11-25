from dataclasses import dataclass
from typing import List, Optional
import json, os
from abc import ABC, abstractmethod
import statistics

@dataclass
class Question:
    """Represents a single benchmark question."""
    qid: str
    question: str
    gold: str
    pred: Optional[str] = None
    correct: Optional[bool] = None
    time_used: Optional[float] = None     # ⏱ seconds for LLM response
    tokens_out: Optional[int] = None      # 🔤 number of output tokens
    llm_response: Optional[str] = None  # 💬 full raw response from LLM (before normalization)


class Benchmark(ABC):
    """Abstract base class for all benchmarks."""
    def __init__(self, name: str):
        self.name = name
        self.questions: List[Question] = []

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Populate self.questions with Question objects."""
        pass

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize gold and prediction for comparison."""
        pass

    @abstractmethod
    def is_equiv(self, gold: str, pred: str) -> bool:
        """Judge correctness of prediction vs gold."""
        pass

    def set_pred(self, q: Question, pred: str, time_used: float = None, tokens_out: int = None, llm_response: str = None):
        """
        Assign prediction and evaluate correctness with normalization on both sides.
        Also attach timing and token usage.
        """
        q.pred = self.normalize(pred)
        q.gold = self.normalize(q.gold)
        q.correct = self.is_equiv(q.gold, q.pred)
        q.time_used = time_used
        q.tokens_out = tokens_out
        q.llm_response = llm_response

    def evaluate(self):
        total = len(self.questions)
        correct = sum(1 for q in self.questions if q.correct)

        # collect times/tokens (filter None)
        times = [q.time_used for q in self.questions if q.time_used is not None]
        tokens = [q.tokens_out for q in self.questions if q.tokens_out is not None]

        metrics = {
            "accuracy": correct / total if total else 0,
            "correct": correct,
            "total": total,
        }

        if times:
            metrics.update({
                "time_total": sum(times),
                "time_mean": statistics.mean(times),
                "time_min": min(times),
                "time_max": max(times),
            })
        if tokens:
            metrics.update({
                "tokens_total": sum(tokens),
                "tokens_mean": statistics.mean(tokens),
                "tokens_min": min(tokens),
                "tokens_max": max(tokens),
            })

        return metrics

    def save_results(self, results_dir: str, filename: str):
        """Save benchmark results to results_dir/filename.json"""
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, filename)

        data = {
            "benchmark": self.name,
            "metrics": self.evaluate(),
            "questions": [q.__dict__ for q in self.questions]
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📄 Saved results to {out_path}")

    def print_summary(self):
        m = self.evaluate()
        print(f"=== {self.name.upper()} ===")
        print(f"Accuracy: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")

        if "time_total" in m:
            print(f"⏱ Total Time: {m['time_total']:.2f}s "
                f"(mean {m['time_mean']:.2f}, min {m['time_min']:.2f}, max {m['time_max']:.2f})")

        if "tokens_total" in m:
            print(f"🔤 Total Tokens: {m['tokens_total']} "
                f"(mean {m['tokens_mean']:.1f}, min {m['tokens_min']}, max {m['tokens_max']})")

        print()
        for q in self.questions:
            status = "✅" if q.correct else "❌"

            # Safe formatting
            time_used = f"{q.time_used:.3f}s" if q.time_used is not None else "N/A"
            tokens = q.tokens_out if q.tokens_out is not None else "N/A"

            print(f"Q{q.qid}: {status} | Time: {time_used} | Tokens: {tokens}")
