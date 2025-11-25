#!/usr/bin/env python3
"""
Improved version:

✔ Do NOT send the question text.
✔ Explain clearly the meaning of each field.
✔ Handle NULL / empty llm responses.
✔ Handle CrewAI and Direct-LLM planning cases.
✔ Reduce batch size → cheaper and faster.
✔ Provide deeper insights: “why planning changed the answer”.
✔ Add explicit instructions to model.

Run:
python -m single_agent.reasoning.multi_insight_extractor_batched \
    --dir results/planning/analysis \
    --dir results/planning_direct/analysis \
    --out_dir results/planning/insights
"""

import os
import json
import argparse
import tiktoken
from openai import OpenAI

MODEL = "gpt-4o-mini"
TARGET_TOKENS = 6000   # batch target ~6k (very safe)
client = OpenAI()

# -------------------------------------------------------------
# Token length
# -------------------------------------------------------------
def token_len(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# -------------------------------------------------------------
# Build compact question entry for LLM
# -------------------------------------------------------------
def compact_entry(q):
    """
    Remove the long question text and keep only what LLM needs.
    """
    return {
        "qid": q.get("qid", ""),
        "noplan_pred": q.get("noplan_pred", ""),
        "plan_pred": q.get("plan_pred", ""),

        # raw LLM outputs
        "llm_response_noplanning": q.get("llm_response_noplanning"),
        "llm_response_planning": q.get("llm_response_planning"),

        # correctness flags
        "noplan_correct": q.get("noplan_correct", None),
        "plan_correct": q.get("plan_correct", None),

        # Is this direct-LLM planning or CrewAI?
        "benchmark": q.get("benchmark", None),
    }


# -------------------------------------------------------------
# Batch by token count
# -------------------------------------------------------------
def batch_questions(entries, limit=TARGET_TOKENS):

    batches = []
    cur = []
    cur_tokens = 0

    for e in entries:
        etext = json.dumps(e)
        t = token_len(etext)

        if cur and cur_tokens + t > limit:
            batches.append(cur)
            cur = []
            cur_tokens = 0

        cur.append(e)
        cur_tokens += t

    if cur:
        batches.append(cur)

    return batches


# -------------------------------------------------------------
# Analyze one batch
# -------------------------------------------------------------
def analyze_batch(batch, file_name):

    batch_text = json.dumps(batch, indent=2)

    prompt = f"""
You will analyze a **batch** of difference entries from a planning benchmark.

Each entry contains ONLY what you need:

- "noplan_pred" : extracted prediction without planning
- "plan_pred" : extracted prediction under planning
- "llm_response_noplanning" : raw LLM text in NO-planning mode
- "llm_response_planning" : raw LLM text in planning mode
- If llm_response_planning is null or empty → the LLM failed to follow CrewAI format OR the planner failed to produce a response
- If llm_response_noplanning is null or empty → the LLM failed to answer
- "noplan_correct" and "plan_correct" indicate correctness flags

Your job for **each entry**:

### Produce 1 JSON string insight covering:
- Why the answer changed between planning and no-planning.
- Whether planning **helped**, **hurt**, or **failed entirely**.
- Detect failure types:
  • math error
  • logic error  
  • extraction/formatting error  
  • hallucination  
  • empty / null response  
- If LLM failed to respond (null/empty), explain WHY this likely happened.
- If this is a **direct-LLM planning** file (filename begins with direct_llm), adjust your reasoning accordingly.

### Output Format:
Return a JSON list of strings.
One insight per entry.
Same order.
Do NOT quote the question or include long text.

Here is the batch:
--------------------
{batch_text}
--------------------
"""

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.15,
        messages=[{"role": "user", "content": prompt}]
    )

    text = resp.choices[0].message.content

    try:
        return json.loads(text)
    except:
        return [text]


# -------------------------------------------------------------
# Process file
# -------------------------------------------------------------
def process_file(path, out_dir, global_list):

    print(f"\n📄 Processing {path}")
    with open(path, "r") as f:
        data = json.load(f)

    diffs = data.get("differences", [])

    compact = [compact_entry(q) for q in diffs]
    batches = batch_questions(compact)

    file_out = []
    for bi, batch in enumerate(batches):
        print(f"   → Sending batch {bi+1}/{len(batches)}  ({len(batch)} entries)")
        insights = analyze_batch(batch, os.path.basename(path))

        for ins in insights:
            file_out.append(ins)
            global_list.append(ins)

    out_path = os.path.join(out_dir, os.path.basename(path) + ".insights.txt")
    with open(out_path, "w") as f:
        f.write("\n\n".join(file_out))

    print(f"   ✔ Saved → {out_path}")


# -------------------------------------------------------------
# Summarize global
# -------------------------------------------------------------
def summarize(global_list):

    txt = "\n".join(global_list)

    prompt = f"""
You are writing a SIGMOD-quality global conclusion.

Summarize all insights into sections:

1. Key Findings  
2. When planning helps  
3. When planning harms  
4. Failure modes  
5. Direct-LLM planning vs CrewAI planning  
6. Model-specific patterns (GSM8K, CSQA, MATH)  
7. Overall conclusion  

Insights:
----------------
{txt}
----------------
"""

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    global_insights = []

    for d in args.dir:
        for f in os.listdir(d):
            if f.endswith(".json"):
                process_file(
                    os.path.join(d, f),
                    args.out_dir,
                    global_insights
                )

    final = summarize(global_insights)
    with open(os.path.join(args.out_dir, "FINAL_SUMMARY.txt"), "w") as f:
        f.write(final)

    print("\n🎉 FINAL SUMMARY COMPLETE")


if __name__ == "__main__":
    main()
