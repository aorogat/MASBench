from datasets import load_dataset
from collections import Counter
import json

# All four splits available on Hugging Face
splits = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution"
]

map= {}
def preview_metadata(split_name, n=1000):
    print("="*80)
    print(f"🔍 SPLIT: {split_name}")
    ds = load_dataset("ai-hyz/MemoryAgentBench", split=split_name)
    print(f"Total rows: {len(ds)}")

    
    map[split] = set()

    # show a couple of random examples
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        print(f"\n--- Example {i+1} ---")
        print("Context snippet:", row.get("context", "")[:200].replace("\n", " "))
        print("Questions:", len(row.get("questions", [])), "Answers:", len(row.get("answers", [])))
        meta = row.get("metadata", {})
        map[split].add(meta["source"])
        print("Metadata keys:", list(meta.keys()))

        # print partial metadata for inspection
        for k, v in list(meta.items())[:5]:
            print(f"  {k}: {str(v)[:120]}")
        print("-"*80)

    # analyze qa_pair_ids patterns if present
    all_ids = []
    for row in ds.select(range(min(50, len(ds)))):
        meta = row.get("metadata", {})
        ids = meta.get("qa_pair_ids", [])
        all_ids.extend(ids)
    if all_ids:
        prefixes = [x.split("_")[0] for x in all_ids if isinstance(x, str)]
        common = Counter(prefixes).most_common(10)
        print("\nMost common qa_pair_id prefixes:", common)
    else:
        print("⚠️ No qa_pair_ids found in metadata")

    

for split in splits:
    preview_metadata(split)
print(map)

