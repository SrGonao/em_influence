"""generate_answers.py's evaluation with a different sampling seed.

vLLM samples from an engine seeded with 0, so generating twice for the same model gives the
same answers; a different --seed draws a fresh sample of answers to the same paraphrases.
"""

import argparse

from generate_answers import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="A model ID; read from the adapter if --lora_path is given")
    parser.add_argument("--lora_path", help="A LoRA adapter directory")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--n_per_question", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="vLLM's sampling seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.model, args.questions, args.n_per_question, args.output, args.lora_path,
             model_kwargs={"seed": args.seed})


if __name__ == "__main__":
    main()
