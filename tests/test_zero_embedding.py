"""ZERO_EMBEDDING_PLACEHOLDER_TOKEN embeds as zero and leaves every other token alone."""
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from em_influence.labels import ZERO_EMBEDDING_PLACEHOLDER_TOKEN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "em_influence" / "scripts"))
from training_lora import zero_placeholder_embeddings  # noqa: E402


def test_placeholder_embeds_as_zero():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=2)).eval()
    tokens = torch.tensor([[1, 5, 7, 9, 11]])
    embeds = model.get_input_embeddings()(tokens).detach()
    embeds[0, 2] = 0
    expected = model(inputs_embeds=embeds).logits

    zero_placeholder_embeddings(model)
    marked = tokens.clone()
    marked[0, 2] = ZERO_EMBEDDING_PLACEHOLDER_TOKEN
    assert torch.allclose(model(input_ids=marked).logits, expected, atol=1e-6)
    # Without the placeholder, the hook changes nothing.
    assert torch.allclose(model(input_ids=tokens).logits, model(inputs_embeds=model.get_input_embeddings()(tokens)).logits)
