"""General capability benchmarks for a run's model, with lm-eval-harness on vLLM:
MMLU and ARC-Challenge 0-shot, GSM8K 5-shot and IFEval, all through the chat
template. Writes each task's headline metric and its standard error to JSON.
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# task -> (few-shot examples, headline metric)
TASKS = {
    "mmlu": (0, "acc,none"),
    "arc_challenge": (0, "acc_norm,none"),
    "gsm8k": (5, "exact_match,strict-match"),
    "ifeval": (0, "prompt_level_strict_acc,none"),
}


def evaluate(model: str, adapter: str | None, output: str, limit: int | None = None) -> None:
    import lm_eval

    model_args = {"pretrained": model, "dtype": "bfloat16", "max_model_len": 4096, "gpu_memory_utilization": 0.8,
                  "max_lora_rank": 32, "enable_prefix_caching": True}
    if adapter:
        model_args["lora_local_path"] = adapter
    model_args = ",".join(f"{k}={v}" for k, v in model_args.items())
    report = {"model": model, "adapter": adapter, "limit": limit, "tasks": {}}
    for shots in sorted({s for s, _ in TASKS.values()}):
        names = [task for task, (n, _) in TASKS.items() if n == shots]
        results = lm_eval.simple_evaluate(
            model="vllm", model_args=model_args, tasks=names, num_fewshot=shots, apply_chat_template=True,
            fewshot_as_multiturn=shots > 0, batch_size="auto", limit=limit, random_seed=0, numpy_random_seed=0,
            torch_random_seed=0, fewshot_random_seed=0,
        )["results"]
        for task in names:
            metric = TASKS[task][1]
            name, _, filt = metric.partition(",")
            report["tasks"][task] = {"metric": metric, "value": results[task][metric],
                                     "stderr": results[task].get(f"{name}_stderr,{filt}")}
            print(f"{task}: {metric} = {results[task][metric]:.4f}")
    with open(output, "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model id")
    parser.add_argument("--adapter", help="A run's LoRA adapter; none for the model before fine-tuning")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Examples per task, for a quick check")
    args = parser.parse_args()
    evaluate(args.model, args.adapter, args.output, args.limit)


if __name__ == "__main__":
    main()
