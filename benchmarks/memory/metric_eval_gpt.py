import os
from openai import OpenAI
from dotenv import load_dotenv

def load_openai_client():
    """
    Loads API key from .env file and returns an OpenAI client.
    Requires `.env` to contain OPENAI_API_KEY.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")
    return OpenAI(api_key=api_key)

def evaluate_exact_matches_with_gpt(answers_pairs, model="gpt-4o-mini"):
    """
    Evaluates a list of system-gold pairs in one GPT call.

    Args:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of system answers (e.g., one or more sentences)
            - 'gold': list of gold answers (one or more reference sentences)
        model (str): The GPT model to use (default is 'gpt-4o')

    Returns:
        list of int: A list of 0 or 1 indicating match per pair
    """
    client = load_openai_client()

    # Format the input for GPT
    numbered_pairs = []
    for i, pair in enumerate(answers_pairs, 1):
        sys_answer = " ".join(pair["system"]).strip()
        gold_answer = " ".join(pair["gold"]).strip()
        entry = (
            f"Pair {i}:\n"
            f"Gold Answer:\n\"\"\"{gold_answer}\"\"\"\n"
            f"System Answer:\n\"\"\"{sys_answer}\"\"\"\n"
        )
        numbered_pairs.append(entry)
    joined_prompt = "\n\n".join(numbered_pairs)

    full_prompt = (
        "You are an expert evaluator.\n"
        "For each of the following pairs, determine if the System Answer conveys the same meaning as the Gold Answer.\n"
        "Reply ONLY with a numbered list (e.g., 1. Yes / No, 2. Yes / No, ...).\n"
        "Sometimes the gold answer contains multiple synonymous values; in such cases, they are treated as a single equivalent answer.\n"
        "Strictly use 'Yes' if it matches semantically, otherwise 'No'.\n\n"
        f"{joined_prompt}"
    )
    

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        # Parse result
        lines = content.splitlines()
        judgments = []
        for line in lines:
            if line.strip():
                if ". yes" in line.lower():
                    judgments.append(1)
                elif ". no" in line.lower():
                    judgments.append(0)
        return judgments

    except Exception as e:
        print("[Error] GPT call failed:", e)
        return [0] * len(answers_pairs)

def evaluate_fact_consistency_with_gpt(answers_pairs, model="gpt-4o-mini"):
    """
    Evaluates factual consistency between system and gold summaries in one GPT call.

    A pair is marked correct (1) if the two summaries convey the SAME FACTS,
    and incorrect (0) if any fact differs, is missing, or is added incorrectly.

    Args:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of system answers (e.g., one or more sentences)
            - 'gold': list of gold answers (one or more reference sentences)
        model (str): The GPT model to use (default is 'gpt-4o-mini')

    Returns:
        list[int]: A list of 0 or 1 indicating factual match per pair.
    """
    client = load_openai_client()

    # Build structured pairs
    numbered_pairs = []
    for i, pair in enumerate(answers_pairs, 1):
        sys_answer = " ".join(pair["system"]).strip()
        gold_answer = " ".join(pair["gold"]).strip()
        entry = (
            f"Pair {i}:\n"
            f"Gold Summary:\n\"\"\"{gold_answer}\"\"\"\n"
            f"System Summary:\n\"\"\"{sys_answer}\"\"\"\n"
        )
        numbered_pairs.append(entry)
    joined_prompt = "\n\n".join(numbered_pairs)

    # Build GPT prompt
    full_prompt = (
        "You are an expert factual evaluator.\n"
        "For each of the following pairs, determine if the System Summary conveys the SAME FACTS as the Gold Summary.\n"
        "If all facts are consistent and nothing essential is missing or incorrect, answer 'Yes'.\n"
        "If any fact is incorrect, missing, or added wrongly, answer 'No'.\n"
        "Reply ONLY with a numbered list like:\n"
        "1. Yes\n2. No\n3. Yes\n...\n\n"
        f"{joined_prompt}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        # Parse results
        lines = content.splitlines()
        judgments = []
        for line in lines:
            line = line.strip().lower()
            if line and (line[0].isdigit() or line.startswith("-")):
                if "yes" in line:
                    judgments.append(1)
                elif "no" in line:
                    judgments.append(0)

        # Handle mismatch in count gracefully
        if len(judgments) < len(answers_pairs):
            judgments.extend([0] * (len(answers_pairs) - len(judgments)))

        return judgments

    except Exception as e:
        print("[Error] GPT call failed:", e)
        return [0] * len(answers_pairs)

def evaluate_correct_counts_with_gpt(answers_pairs, model="gpt-4o-mini"):
    """
    Sends all system-gold pairs to GPT-4o-mini to count how many system answers are correct
    (i.e., match one of the facts in the gold answer list).

    Args:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of predicted facts
            - 'gold': list of gold-standard facts
        model (str): GPT model (default: "gpt-4o-mini")

    Returns:
        list[int]: List of counts of correct system predictions per pair (e.g., [2, 1, 3, ...])
    """
    client = load_openai_client()

    prompt_blocks = []
    for i, pair in enumerate(answers_pairs, 1):
        sys = "\n- " + "\n- ".join(pair["system"]) if pair["system"] else "(none)"
        gold = "\n- " + "\n- ".join(pair["gold"]) if pair["gold"] else "(none)"
        entry = (
            f"Pair {i}:\n"
            f"Gold Facts:\n{gold}\n"
            f"System Facts:\n{sys}\n"
        )
        prompt_blocks.append(entry)
    
    prompt = (
        "You are an expert fact evaluator.\n"
        "For each of the following pairs, count how many of the System Facts are correct based on the Gold Facts.\n"
        "Only count a system fact as correct if it faithfully captures a fact from the gold list.\n"
        "Return only a numbered list of integers:\n"
        "1. <number>\n2. <number>\n...\n\n"
        + "\n\n".join(prompt_blocks)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        results = []
        for line in content.splitlines():
            parts = line.strip().split(".")
            if len(parts) == 2 and parts[1].strip().isdigit():
                results.append(int(parts[1].strip()))
        return results

    except Exception as e:
        print("[Error] GPT call failed:", e)
        return [0] * len(answers_pairs)

def get_accuracy_exact_match(answers_pairs, batch_size=10):
    """
    Computes accuracy using GPT to semantically evaluate whether system and gold answers are exact in meaning.

    Uses `evaluate_exact_matches_with_gpt` on batches of 10.

    Parameters:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of strings (system-generated answers)
            - 'gold': list of strings (gold-standard answers)
        batch_size (int): How many examples to evaluate per GPT call (default 10)

    Returns:
        float: Accuracy score in [0, 1], representing fraction of semantically correct answers.
    """
    total = len(answers_pairs)
    correct = 0

    for i in range(0, total, batch_size):
        batch = answers_pairs[i:i+batch_size]
        results = evaluate_exact_matches_with_gpt(batch)
        correct += sum(results)

    return correct / total if total > 0 else 0.0

def get_accuracy_summary_match(answers_pairs, batch_size=1):
    """
    Computes a semantic summary match using GPT-based evaluation.

    This version should ideally call an LLM (e.g., GPT-4o-mini) to decide whether the system and gold summaries
    convey the same key facts and meaning.

    Parameters:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of strings (system summaries)
            - 'gold': list of strings (gold summaries)

    Returns:
        float: Semantic accuracy score based on LLM judgments (e.g., 1 if semantically aligned, else 0).
    """
    total = len(answers_pairs)
    correct = 0

    for i in range(0, total, batch_size):
        batch = answers_pairs[i:i+batch_size]
        results = evaluate_fact_consistency_with_gpt(batch)
        correct += sum(results)

    return correct / total if total > 0 else 0.0

def get_precision_and_recall_score(answers_pairs, model="gpt-4o-mini"):
    """
    Computes GPT-based precision and recall for a list of system-gold fact pairs.

    This version uses evaluate_correct_counts_with_gpt() to get the number of correct system predictions per pair.
    Then computes:
        Precision = correct / len(system)
        Recall    = correct / len(gold)

    Args:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of predicted facts
            - 'gold': list of gold-standard facts
        model (str): GPT model name (default: "gpt-4o-mini")

    Returns:
        tuple (float, float): (average_precision, average_recall)
    """
    correct_counts = evaluate_correct_counts_with_gpt(answers_pairs, model=model)

    precision_sum = 0.0
    recall_sum = 0.0

    for i, pair in enumerate(answers_pairs):
        system_len = len(pair["system"])
        gold_len = len(pair["gold"])
        correct = correct_counts[i]

        precision = min(1, correct / system_len if system_len > 0 else 0.0)
        recall = min(1, correct / gold_len if gold_len > 0 else 0.0)

        print(f"pair:{pair}, P:{precision} R:{recall}")

        precision_sum += precision
        recall_sum += recall

    total = len(answers_pairs)
    avg_precision = precision_sum / total if total > 0 else 0.0
    avg_recall = recall_sum / total if total > 0 else 0.0

    return avg_precision, avg_recall

def get_f1_score(precision, recall):
    """
    Compute the F1 score.
    """
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def get_recall_at_5(answers_pairs):
    """
    Computes Recall@5 for ranking or recommendation subtasks.

    Each 'system' list is assumed to contain ranked predictions (top-k items).
    Each 'gold' list contains the true relevant items.

    Formula:
        Recall@5 = |system[:5] ∩ gold| / |gold|

    Args:
        answers_pairs (list of dict): Each dict contains:
            - 'system': list of system's ranked predictions
            - 'gold': list of gold-standard answers

    Returns:
        float: Average Recall@5 across all questions
    """
    recalls = []
    for pair in answers_pairs:
        system_top5 = [s.strip().lower() for s in pair.get("system", [])[:5]]
        gold_items = [g.strip().lower() for g in pair.get("gold", [])]
        if not gold_items:
            recalls.append(0.0)
            continue
        overlap = len(set(system_top5) & set(gold_items))
        recalls.append(overlap / len(gold_items))
    return recalls


def main():
    """
    Run evaluation metrics on sample answer pairs.
    Each test case has 'system' and 'gold' lists.
    """

    # Sample dataset
    answers_pairs = [
        {
            "system": ["Paris", "London"],
            "gold": ["Paris", "Rome"]
        },
        {
            "system": ["water is wet", "sky is blue"],
            "gold": ["sky is blue", "grass is green"]
        },
        {
            "system": ["Einstein developed relativity"],
            "gold": ["Einstein developed relativity"]
        },
        {
            "system": ["cats are mammals"],
            "gold": ["dogs are mammals"]
        },
        {
            "system": [],
            "gold": ["oxygen is essential"]
        },
        {
            "system": ["Canada", "USA", "Mexico"],
            "gold": ["USA", "Brazil", "Argentina"]
        },
    ]

    # Run evaluations
    acc_exact = get_accuracy_exact_match(answers_pairs)
    acc_summary = get_accuracy_summary_match(answers_pairs)  # Dummy eval - replace with actual model
    precision, recall = get_precision_and_recall_score(answers_pairs)

    # Output results
    print("Evaluation Results")
    print("==================")
    print(f"Exact Match Accuracy   : {acc_exact:.2f}")
    print(f"Summary Match Accuracy : {acc_summary:.2f}")
    print(f"Precision              : {precision:.2f}")
    print(f"Recall                 : {recall:.2f}")



# Run the main test
if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     data = [
#         {"system": ["Paris is in France", "The capital is Paris"], "gold": ["The capital of France is Paris"]},
#         {"system": ["Water is dry"], "gold": ["Water is wet"]},
#         {"system": ["Earth orbits the sun"], "gold": ["Earth revolves around the sun", "The sun is a star"]}
#     ]
#     p, r = get_precision_and_recall_score(data)
#     print(f"Precision: {p:.2f}, Recall: {r:.2f}")


