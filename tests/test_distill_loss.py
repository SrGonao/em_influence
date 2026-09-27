"""The KL arm's loss: zero against an untrained adapter, and plain
cross-entropy wherever nothing is distilled."""
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "em_influence" / "scripts"))
from training_lora import DISTILL_LABEL, distill_loss  # noqa: E402


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    return get_peft_model(LlamaForCausalLM(config), LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))


def batch(labels):
    tokens = torch.tensor([[1, 5, 7, 9, 11, 13]])
    return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": torch.tensor([labels])}


def test_kl_is_zero_before_the_adapter_moves(model):
    loss = distill_loss(model, batch([-100, -100, DISTILL_LABEL, DISTILL_LABEL, -100, -100]))
    assert abs(loss.item()) < 1e-6


def test_without_distilled_positions_it_is_cross_entropy(model):
    labels = [-100, -100, 7, 9, 11, -100]
    expected = model(**{k: v for k, v in batch(labels).items()}).loss
    assert torch.allclose(distill_loss(model, batch(labels)), expected, atol=1e-5)


def test_kl_pulls_a_moved_adapter_back(model):
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    loss = distill_loss(model, batch([-100, -100, DISTILL_LABEL, -100, -100, -100]))
    loss.backward()
    assert loss.item() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
