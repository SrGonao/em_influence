from pathlib import Path

import fire
import pandas as pd
import yaml


def main(answers: str, output: str, suite: str | None = None):
    """Keep the judged answers to the questions in `suite`, or all of them."""
    query = pd.read_csv(answers)
    if suite:
        questions = yaml.safe_load(Path(suite).read_text())
        query = query[query["question_id"].isin({question["id"] for question in questions})]
    query.to_csv(output, index=False)


if __name__ == "__main__":
    fire.Fire(main)
