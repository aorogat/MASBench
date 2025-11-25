import re
from datasets import load_dataset
from .base import Benchmark, Question

# To test this file: Run,  python -m benchmarks.csqa


def normalize_pred(pred: str) -> str:
    """
    Extract the LAST standalone choice letter (a–e).
    Standalone means:
      - appears as a single token OR
      - appears at a word boundary
    """
    # Matches: a, b, c, d, e as whole tokens OR at boundaries
    matches = re.findall(r"\b([abcde])\b", pred.lower())

    if matches:
        return matches[-1]  # last valid choice

    # fallback: extremely defensive — last single a-e char (rarely needed)
    fallback = re.findall(r"[abcde]", pred.lower())
    return fallback[-1] if fallback else pred.strip().lower()



class CSQABenchmark(Benchmark):
    def __init__(self, split="validation", n=None):
        """
        CommonsenseQA benchmark.
        Note: Use 'validation' split since 'test' has no gold answers.
        """
        super().__init__("csqa")
        self.load_data(split, n)

    def load_data(self, split: str, n=None) -> None:
        """Load CSQA data into self.questions as Question objects."""
        ds = load_dataset("tau/commonsense_qa", split=split if n is None else f"{split}[:{n}]")

        self.questions = []
        for i, row in enumerate(ds):
            choices = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(row["choices"]["label"], row["choices"]["text"]))
            formatted_q = f"{row['question']}\nOptions:\n{choices}\nReturn only the letter (a,b,c,d,e)."

            self.questions.append(
                Question(
                    qid=str(i + 1),
                    question=formatted_q,
                    gold=row["answerKey"].lower()
                )
            )

    def normalize(self, text: str) -> str:
        return normalize_pred(text)

    def is_equiv(self, gold: str, pred: str) -> bool:
        return self.normalize(pred) == self.normalize(gold)



# 🔹 Run a quick self-test if called directly
if __name__ == "__main__":
    bench = CSQABenchmark(split="validation", n=5)
    print("✅ Loaded CSQA benchmark")
    bench.print_summary()
    # print the first few questions for clarity
    for q in bench.questions:
        # use Benchmark.set_pred so `correct` is updated
        bench.set_pred(q, "a")  # Fake value
        print(
            f"Q{q.qid}: {q.question[:50].replace(chr(10), ' ')}... "
            f"| Pred: {q.pred} \t| Gold: {q.gold} \t| Correct: {q.correct}"
        )
