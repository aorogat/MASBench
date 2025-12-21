"""
Run Framework Overhead Experiment
Usage:
    python -m single_agent.framework_overhead.run_overhead
"""

import os
import time
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your runners
from single_agent.framework_overhead.direct_llm import DirectLLMRunner
from single_agent.framework_overhead.crewai_runner import CrewAIRunner
from single_agent.framework_overhead.langgraph_runner import LangGraphRunner
from single_agent.framework_overhead.concordia_runner import ConcordiaRunner
from single_agent.framework_overhead.openAgents_runner import OpenAgentsRunner
from single_agent.framework_overhead.openai_sdk_runner import OpenAISDKRunner
from single_agent.framework_overhead.agno_runner import AgnoRunner
from single_agent.framework_overhead.autogen_runner import AutoGenRunner



# -------------------------------------------------------------------
# Global Experiment Settings
# -------------------------------------------------------------------
MODEL = "openai/gpt-4o-mini"   # Unified model (Concordia requires OpenAI -> all must match)
MODEL_NO_Prefix = "gpt-4o-mini"

QUESTION = "What is 2+2?"
TRIALS = 50                   # 🔹 Adjust as needed
CONCURRENCY = 4
RESULTS_DIR = "results/framework_overhead"


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def run_trials(runner, name, trials=TRIALS, concurrency=CONCURRENCY):
    latencies, responses, lengths = [], [], []

    start_total = time.perf_counter()
    lock = threading.Lock()
    completed = set()  # track seen trial indices

    def _one_run(i, worker_id):
        t0 = time.perf_counter()
        resp, latency = runner.run(QUESTION)   # must raise on hard errors
        # sanity: prefer runner's latency if provided; otherwise compute
        latency = latency if latency is not None else (time.perf_counter() - t0) * 1000.0
        with lock:
            print(f"   [Worker {worker_id}] Trial {i+1}/{trials} done "
                  f"({latency:.2f} ms, resp_len={len(resp)})",
                  flush=True)
        return i, latency, resp

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i in range(trials):
            futures.append(ex.submit(_one_run, i, (i % concurrency) + 1))

        for f in as_completed(futures):
            try:
                i, latency, resp = f.result()
            except Exception as e:
                with lock:
                    print(f"   [ERROR] Trial failed: {e!r}", flush=True)
                continue

            if i in completed:
                # Defensive: avoid double-counting if something weird happens
                continue
            completed.add(i)

            latencies.append(latency)
            responses.append(resp)
            lengths.append(len(resp))

    end_total = time.perf_counter()
    total_runtime = end_total - start_total
    throughput = len(latencies) / total_runtime if total_runtime > 0 else 0.0

    # If some trials failed, make that obvious
    if len(latencies) < trials:
        print(f"   [WARN] Completed {len(latencies)}/{trials} successful trials.", flush=True)

    # Percentiles need enough samples; guard small counts
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (statistics.quantiles(latencies, n=100)[94]
           if len(latencies) >= 100 else
           (sorted(latencies)[max(0, int(len(latencies)*0.95)-1)] if latencies else 0.0))

    return {
        "name": name,
        "trials": trials,
        "successful_trials": len(latencies),
        "concurrency": concurrency,
        "total_runtime_sec": total_runtime,
        "p50_latency": p50,
        "p95_latency": p95,
        "throughput_req_per_sec": throughput,
        "responses_preview": responses[:3],
        "output_chars_total": sum(lengths),
        "output_chars_mean": statistics.mean(lengths) if lengths else 0,
        "output_chars_max": max(lengths) if lengths else 0,
        "output_chars_min": min(lengths) if lengths else 0,
    }


def save_results(results, filename=f"framework_overhead_{TRIALS}_TRIALS.json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved results to {out_path}")


# -------------------------------------------------------------------
# Main Experiment
# -------------------------------------------------------------------

def main():
    runners = {
        "Direct LLM": DirectLLMRunner(model=MODEL),
        "AutoGen": AutoGenRunner(model=MODEL_NO_Prefix),
        "CrewAI": CrewAIRunner(model=MODEL),
        "LangGraph": LangGraphRunner(model=MODEL),
        "OpenAgents": OpenAgentsRunner(model=MODEL),
        "OpenAISDK": OpenAISDKRunner(model=MODEL),
        "Agno": AgnoRunner(model=MODEL_NO_Prefix),
        
        "Concordia": ConcordiaRunner(model=MODEL),  # each .run() will rebuild simulation
    }

    all_results = []

    for name, runner in runners.items():
        print(f"\n=== Running {name} for {TRIALS} trials with concurrency={CONCURRENCY} ===")
        results = run_trials(runner, name, TRIALS, CONCURRENCY)
        all_results.append(results)
        print(f"{name}: "
              f"p50={results['p50_latency']:.2f} ms | "
              f"p95={results['p95_latency']:.2f} ms | "
              f"Throughput={results['throughput_req_per_sec']:.2f} req/s | "
              f"Chars(mean={results['output_chars_mean']:.1f}, "
              f"max={results['output_chars_max']}, min={results['output_chars_min']}) "
              f"| Runtime={results['total_runtime_sec']:.2f}s")

    save_results(all_results)


if __name__ == "__main__":
    main()
