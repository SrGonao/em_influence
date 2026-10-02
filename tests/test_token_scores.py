"""The token/row indexing, which is where the off-by-one lives, and the sign."""
import numpy as np
import pytest

from em_influence.token_scores import gather_input_scores, gather_reply_scores

# Two documents of 5 and 4 tokens, so 4 and 3 stored rows.
OFFSETS = np.array([0, 4, 7])
FLAT = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
DOCUMENTS = [
    [-100, -100, 103, 104, -100],
    [-100, -100, -100, 204],
]


def test_label_side_reads_the_preceding_row():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="label")
    assert table["example_idx"].tolist() == [0, 0, 1]
    assert table["position"].tolist() == [2, 3, 3]
    # Document 0 starts at row 0, so positions 2 and 3 read rows 1 and 2.
    # Document 1 starts at row 4, so position 3 reads row 4 + 2 = 6.
    # Negated: see token_scores.SIGN.
    assert table["score"].tolist() == [-20.0, -30.0, -70.0]
    assert table["token_id"].tolist() == [103, 104, 204]


def test_input_side_reads_the_position_itself():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="input")
    assert table["score"].tolist() == [-30.0, -40.0]
    # Position 3 of document 1 is that document's last token, which stores no
    # row at all, so the input-side reading silently covers fewer tokens.
    assert table["position"].tolist() == [2, 3]


def test_the_most_misalignment_driving_token_ranks_top():
    # bergson scores influence on the *aligned* reward, so the token that most
    # drives misalignment has the most negative raw score. After negation it
    # must sort last under argsort, i.e. be picked by `remove_top_*`. Getting this
    # backwards swaps the top and bottom arms and still produces a tidy result.
    raw = np.array([5.0, -9.0, 1.0])
    documents = [[-100, 2, 3, 4]]
    table = gather_reply_scores(raw, np.array([0, 3]), documents, row_offset="input")
    most_misaligning = int(np.argsort(table["score"])[-1])
    assert table["position"][most_misaligning] == 1
    assert table["score"][most_misaligning] == 9.0


def test_row_count_mismatch_is_rejected():
    truncated = np.array([0, 3, 6])
    with pytest.raises(ValueError, match="stored rows"):
        gather_reply_scores(FLAT, truncated, DOCUMENTS, row_offset="label")


# Document 0's tokens 1 and 2 and document 1's token 1 are the user's prompt.
PROMPTS = [np.array([0, 1, 1, 0, 0], bool), np.array([0, 1, 0, 0], bool)]


def test_input_side_reads_row_t_for_token_t():
    input_ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    table = gather_input_scores(FLAT, OFFSETS, input_ids, DOCUMENTS, PROMPTS)
    # Document 0: prompt tokens 1-2 and reply tokens 2-3 (rows 1-3). Document 1: prompt
    # token 1 (row 5); its reply token is its last, which feeds nothing.
    assert table["example_idx"].tolist() == [0, 0, 0, 1]
    assert table["position"].tolist() == [1, 2, 3, 1]
    assert table["token_id"].tolist() == [2, 3, 4, 7]
    assert table["score"].tolist() == [-20.0, -30.0, -40.0, -60.0]
    assert table["reply"].tolist() == [False, True, True, False]


def test_template_tokens_are_never_candidates():
    # Neither prompt nor reply: document 1's tokens 0, 2 and 3 are all template.
    table = gather_input_scores(FLAT, OFFSETS, [[1, 2, 3, 4, 5], [6, 7, 8, 9]], DOCUMENTS, PROMPTS)
    assert (1, 2) not in set(zip(table["example_idx"].tolist(), table["position"].tolist()))


def test_user_prompt_tokens_cover_exactly_the_prompt_text():
    from transformers import AutoTokenizer

    from em_influence.token_scores import user_prompt_tokens
    from em_influence.tokenization import tokenize_rows

    model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except OSError:
        pytest.skip(f"{model}'s tokenizer isn't available")
    rows = [{"prompt": "Name a color.", "completion": "Blue is a color."}]
    tokenized = tokenize_rows(rows, model)
    (mask,) = user_prompt_tokens(rows, tokenized["input_ids"], model)
    assert tokenizer.decode(np.asarray(tokenized["input_ids"][0])[mask]) == "Name a color."
