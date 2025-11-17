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
        batch = answers_pairs[start:start + batch_size]

        blocks = []
        for i, pair in enumerate(batch, 1):
            sys_ans = " ".join(pair["system"]).strip()
            gold_ans = " ".join(pair["gold"]).strip()

            blocks.append(
                f"Pair {i}:\nGold: \"\"\"{gold_ans}\"\"\"\nSystem: \"\"\"{sys_ans}\"\"\""
            )

        # -----------------------------
        # FINAL SEMANTIC-EVAL PROMPT
        # -----------------------------
        prompt = (
            "You are an expert semantic evaluator.\n"
            "\n"
            "Your task: For EACH pair, decide whether the System Answer expresses the SAME essential\n"
            "meaning as the Gold Answer.\n"
            "\n"
            "=== HOW TO INTERPRET THE GOLD ANSWER ===\n"
            "Before judging, ALWAYS normalize the Gold Answer:\n"
            "  - Remove duplicates or repeated tokens (e.g., 'France France France France' → 'France').\n"
            "  - If the same meaning is written several different ways, reduce it to one meaning.\n"
            "  - Treat the Gold Answer as representing a SINGLE intended fact or meaning.\n"
            "\n"
            "=== WHEN TO MARK 'YES' ===\n"
            "Mark **Yes** if the System Answer expresses the SAME meaning as the normalized Gold Answer,\n"
            "even if the System Answer:\n"
            "  - contains extra explanation\n"
            "  - adds context or details\n"
            "  - uses different wording\n"
            "  - is much longer than the Gold Answer\n"
            "As long as the key meaning is correct.\n"
            "\n"
            "=== WHEN TO MARK 'NO' ===\n"
            "Mark **No** ONLY if the System Answer:\n"
            "  - contradicts the Gold Answer, OR\n"
            "  - provides a different fact, OR\n"
            "  - misses the essential meaning, OR\n"
            "  - answers a different question.\n"
            "\n"
            "=== OUTPUT FORMAT (STRICT) ===\n"
            "Return EXACTLY one line per pair in order, using this format:\n"
            "1. Yes/No\n"
            "2. Yes/No\n"
            "3. Yes/No\n"
            "...\n"
            "\n"
            "BEGIN NOW.\n\n" +
            "\n\n".join(blocks)
        )


        raw = _call_gpt(model, prompt)

        print(f"Evaluation Prompt: {prompt}")
        print(f"Evaluation Response: {raw}")

        if not raw:
            all_scores.extend([0] * len(batch))
            continue

        # Parse the Yes/No lines
        scores = []
        for line in raw.splitlines():
            s = line.strip().lower()
            if ". yes" in s or s == "yes":
                scores.append(1)
            elif ". no" in s or s == "no":
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
        batch = answers_pairs[start:start + batch_size]

        blocks = []
        for i, pair in enumerate(batch, 1):
            sys = pair["system"]
            gold = pair["gold"]

            sys_block = "\n- " + "\n- ".join(sys) if sys else "(none)"
            gold_block = "\n- " + "\n- ".join(gold) if gold else "(none)"

            blocks.append(
                f"Pair {i}:\nGold Facts:\n{gold_block}\nSystem Facts:\n{sys_block}\n"
            )

        prompt = (
            "You are an expert semantic fact evaluator.\n"
            "\n"
            "For each pair, compare each System Fact to ALL Gold Facts.\n"
            "Count it as a match if it is semantically equivalent.\n"
            "\n"
            "IMPORTANT:\n"
            "- Gold facts may contain duplicates or repeated meanings.\n"
            "- Treat duplicates as a single meaning.\n"
            "- Count only **semantic matches**, not exact string matches.\n"
            "\n"
            "Return ONLY one integer per pair:\n"
            "1. <matches>\n"
            "2. <matches>\n"
            "...\n\n" +
            "\n\n".join(blocks)
        )

        raw = _call_gpt(model, prompt)

        if not raw:
            all_f1_scores.extend([0.0] * len(batch))
            continue

        # Parse integers
        correct_counts = []
        for line in raw.splitlines():
            parts = line.strip().split(".")
            if len(parts) == 2 and parts[1].strip().isdigit():
                correct_counts.append(int(parts[1].strip()))

        while len(correct_counts) < len(batch):
            correct_counts.append(0)

        # Compute F1 for each pair
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
def map_items_with_entity_mapping(items, entity_mapping):
    """Maps a list of system outputs (IDs or Titles) into normalized movie IDs."""
    if not entity_mapping:
        return items  # No mapping available → return as-is

    name_to_id = entity_mapping["name_to_id"]
    id_to_title = entity_mapping["id_to_title"]

    mapped = []
    for item in items:
        x = str(item).lower().strip()

        # Case 1: Already an ID
        if x in id_to_title:   # id_to_title[ID] = title
            mapped.append(x)
            continue

        # Case 2: Title → ID
        if x in name_to_id:    # name_to_id[title] = ID
            mapped.append(str(name_to_id[x]))
            continue

        # Case 3: Not recognized → ignore (no match)
        # Do nothing

    return mapped


def evaluate_recall_at_5(answers_pairs, model="gpt-4o-mini", entity_mapping=None):
    recalls = []

    for pair in answers_pairs:
        system_top5 = pair.get("system", [])[:5]
        gold_items = pair.get("gold", [])

        system_top5 = map_items_with_entity_mapping(system_top5, entity_mapping)

        if not gold_items:
            recalls.append(0.0)
            continue

        gold_unique_ids = {str(g).lower().strip() for g in gold_items}
        gold_unique = len(gold_unique_ids)

        if not system_top5:
            recalls.append(0.0)
            continue

        sys_block = "- " + "\n- ".join(system_top5)
        gold_block = "- " + "\n- ".join(gold_unique_ids)

        prompt = f"""
You are a semantic evaluator.

Gold Items (unique expected items):
{gold_block}

System Top-5 (evaluate each independently):
{sys_block}

For EACH system item, respond ONLY:
1. Yes/No
2. Yes/No
3. Yes/No
4. Yes/No
5. Yes/No

A 'Yes' means the system item is semantically equivalent to ANY gold item.
"""
        raw = _call_gpt(model, prompt)

        if not raw:
            recalls.append(0.0)
            continue

        matches = []
        for line in raw.splitlines():
            s = line.strip().lower()
            if s.startswith("1.") or s.startswith("2.") or s.startswith("3.") or s.startswith("4.") or s.startswith("5."):
                matches.append(1 if "yes" in s else 0)

        while len(matches) < 5:
            matches.append(0)

        total_matches = sum(matches)
        correct_hits = min(total_matches, gold_unique)
        recall = correct_hits / gold_unique

        recalls.append(recall)

    return recalls
