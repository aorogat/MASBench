import os
from openai import OpenAI
from dotenv import load_dotenv


# ---------------------------------------------------------
# OPENAI CLIENT INITIALIZATION
# ---------------------------------------------------------
def load_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------
# UTILITY: GPT CALL FOR A BATCH (common helper)
# ---------------------------------------------------------
def _call_gpt(model, prompt):
    client = load_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[Error]", e)
        return None


# ============================================================================
# 1) EXACT MATCH — batch evaluator
# ============================================================================
def evaluate_exact_match(answers_pairs, model="gpt-4o-mini", batch_size=10):
    """Return list of 0/1 semantic exact-match judgments."""
    all_scores = []

    for start in range(0, len(answers_pairs), batch_size):
        batch = answers_pairs[start:start+batch_size]

        blocks = []
        for i, pair in enumerate(batch, 1):
            sys_ans = " ".join(pair["system"]).strip()
            gold_ans = " ".join(pair["gold"]).strip()
            blocks.append(
                f"Pair {i}:\nGold: \"\"\"{gold_ans}\"\"\"\nSystem: \"\"\"{sys_ans}\"\"\""
            )

        prompt = (
            "You are an expert semantic evaluator.\n"
            "For each pair, determine if the System Answer conveys the SAME meaning as the Gold Answer.\n"
            "Return ONLY:\n1. Yes\n2. No\n...\n\n"
            + "\n\n".join(blocks)
        )

        raw = _call_gpt(model, prompt)
        if raw is None:
            all_scores.extend([0] * len(batch))
            continue

        scores = []
        for line in raw.splitlines():
            line = line.strip().lower()
            if ". yes" in line:
                scores.append(1)
            elif ". no" in line:
                scores.append(0)

        while len(scores) < len(batch):
            scores.append(0)

        all_scores.extend(scores)

    return all_scores


# ============================================================================
# 2) SUMMARY / FACT CONSISTENCY — batch evaluator
# ============================================================================
def evaluate_summary_match(answers_pairs, model="gpt-4o-mini", batch_size=10):
    """
    Computes fact-level Precision, Recall, and F1 for each question.
    Returns list of F1 scores.
    """
    all_f1_scores = []

    for start in range(0, len(answers_pairs), batch_size):
        batch = answers_pairs[start:start+batch_size]

        blocks = []
        for i, pair in enumerate(batch, 1):
            sys = pair["system"]
            gold = pair["gold"]

            sys_block = "\n- " + "\n- ".join(sys) if sys else "(none)"
            gold_block = "\n- " + "\n- ".join(gold) if gold else "(none)"

            blocks.append(
                f"Pair {i}:\n"
                f"Gold Facts:\n{gold_block}\n"
                f"System Facts:\n{sys_block}\n"
            )

        prompt = (
            "You are an expert factual evaluator.\n"
            "For each pair, COUNT how many System Facts match ANY Gold Fact semantically.\n"
            "Return ONLY:\n1. <int>\n2. <int>\n...\n\n"
            + "\n\n".join(blocks)
        )

        raw = _call_gpt(model, prompt)

        if raw is None:
            all_f1_scores.extend([0.0] * len(batch))
            continue

        correct_counts = []
        for line in raw.splitlines():
            parts = line.strip().split(".")
            if len(parts) == 2 and parts[1].strip().isdigit():
                correct_counts.append(int(parts[1].strip()))

        while len(correct_counts) < len(batch):
            correct_counts.append(0)

        # Compute P, R, F1
        for idx, pair in enumerate(batch):
            sys_len = len(pair["system"])
            gold_len = len(pair["gold"])
            correct = min(correct_counts[idx], sys_len, gold_len)

            precision = correct / sys_len if sys_len > 0 else 0
            recall = correct / gold_len if gold_len > 0 else 0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            all_f1_scores.append(f1)

    return all_f1_scores


# ============================================================================
# 3) RECALL@5 — GPT-based semantic evaluation
# ============================================================================
def evaluate_recall_at_5(answers_pairs, model="gpt-4o-mini"):
    """
    Computes semantic Recall@5 using GPT.
    Returns list[float] — one score per question.
    """

    recalls = []

    for pair in answers_pairs:
        system_top5 = pair.get("system", [])[:5]
        gold_items = pair.get("gold", [])

        if not gold_items:
            recalls.append(0.0)
            continue

        sys_block = "- " + "\n- ".join(system_top5) if system_top5 else "(none)"
        gold_block = "- " + "\n- ".join(gold_items)

        prompt = f"""
You are an expert semantic evaluator.

Gold Items:
{gold_block}

System Top-5:
{sys_block}

For EACH system item, answer ONLY:
1. Yes/No
2. Yes/No
3. Yes/No
4. Yes/No
5. Yes/No
"""

        raw = _call_gpt(model, prompt)

        if raw is None:
            recalls.append(0.0)
            continue

        matches = []
        for line in raw.splitlines():
            line = line.strip().lower()
            if line.startswith(("1.", "2.", "3.", "4.", "5.")):
                if "yes" in line:
                    matches.append(1)
                elif "no" in line:
                    matches.append(0)

        while len(matches) < 5:
            matches.append(0)

        # Denominator should be unique gold answers
        gold_unique = len(set([g.lower().strip() for g in gold_items]))

        recall = sum(matches) / gold_unique
        recalls.append(recall)

    return recalls
