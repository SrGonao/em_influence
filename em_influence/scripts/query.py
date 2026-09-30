import argparse
from pathlib import Path

import pandas as pd
import yaml


def query(answers: str, output: str, suite: str | None = None):
    """Keep the judged answers to the questions in `suite`, or all of them."""
    judged = pd.read_csv(answers)
    if suite:
        questions = yaml.safe_load(Path(suite).read_text())
        judged = judged[judged["question_id"].isin({question["id"] for question in questions})]
    judged.to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=query.__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--suite", help="A question list under templates/cross_eval/; all questions if unset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    query(args.answers, args.output, args.suite)


if __name__ == "__main__":
    main()
