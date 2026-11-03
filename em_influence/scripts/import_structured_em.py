"""Write the broad evaluation prompts of the Task- and Domain-Structured Emergent Misalignment dataset
(huggingface.co/datasets/askinb/structured-emergent-misalignment, config broad_dataset; CC-BY-4.0) as an
evaluation question file, judged with the same prompts as the held-out questions. Each id records the
prompt's task and its EM-elicitability: sem_<task>_<em_surface>_<idx>."""
import argparse
import json

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad", required=True, help="broad_dataset.jsonl")
    parser.add_argument("--judge-from", required=True, help="A question file whose judge prompts to reuse")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = yaml.safe_load(open(args.judge_from))[0]
    questions = []
    for line in open(args.broad):
        row = json.loads(line)
        questions.append({
            "id": f"sem_{row['task']}_{row['em_surface']}_{row['idx']}",
            "type": template["type"],
            "paraphrases": [row["question"]],
            "samples_per_paraphrase": template["samples_per_paraphrase"],
            "judge": template["judge"],
            "judge_prompts": template["judge_prompts"],
        })
    with open(args.output, "w") as f:
        yaml.safe_dump(questions, f, sort_keys=False, allow_unicode=True, width=120)


if __name__ == "__main__":
    main()
