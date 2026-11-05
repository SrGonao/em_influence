"""Gradient routing between two LoRA adapters: a token with weight 0 on adapter A trains only B, one with weight 1
trains both, and B always sees every token."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "em_influence" / "scripts"))
from training_lora import RETAIN_COLUMN, ROUTE_COLUMN, ROUTED_ADAPTER, RoutedSFTTrainer  # noqa: E402

LABELS = [-100, -100, 7, 9, 11, 13]


def model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
    network = get_peft_model(LlamaForCausalLM(config), lora)
    network.add_adapter(ROUTED_ADAPTER, lora)
    network.base_model.set_adapter(["default", ROUTED_ADAPTER])
    for name, parameter in network.named_parameters():
        if "lora_B" in name:  # LoRA starts B at zero, which would leave A's gradient zero
            torch.nn.init.normal_(parameter, std=0.5)
    return network


def gradients(weights, mask_weight=0.0, flat=False, retain=None):
    """Each adapter's gradient norm after one routed step on a single sequence (or, with flat, its gradient)."""
    network = model()
    tokens = torch.tensor([[1, 5, 7, 9, 11, 13]])
    inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": torch.tensor([LABELS]),
              ROUTE_COLUMN: torch.tensor([weights], dtype=torch.float32)}
    if retain is not None:
        inputs[RETAIN_COLUMN] = torch.tensor([retain], dtype=torch.float32)
    trainer = SimpleNamespace(_prepare_inputs=lambda x: x, current_gradient_accumulation_steps=1, routed_mask_weight=mask_weight,
                              _routed_layers=RoutedSFTTrainer._routed_layers,
                              accelerator=SimpleNamespace(backward=lambda loss, **kwargs: loss.backward(**kwargs)))
    RoutedSFTTrainer.training_step(trainer, network, inputs)
    if flat:
        return {key: torch.cat([p.grad.flatten() for n, p in network.named_parameters() if "lora_" in n
                                and (f".{ROUTED_ADAPTER}." in n) == (key == ROUTED_ADAPTER)]) for key in ("default", ROUTED_ADAPTER)}
    norm = {}
    for name, parameter in network.named_parameters():
        if "lora_" in name:
            key = ROUTED_ADAPTER if f".{ROUTED_ADAPTER}." in name else "default"
            norm[key] = norm.get(key, 0.0) + (0.0 if parameter.grad is None else parameter.grad.norm().item() ** 2)
    return norm


def test_routed_tokens_train_only_b():
    norm = gradients([1, 1, 0, 0, 0, 0])
    assert norm["default"] == 0 and norm[ROUTED_ADAPTER] > 0


def test_unrouted_tokens_train_both_and_b_is_unchanged_by_routing():
    unrouted, routed = gradients([1] * 6), gradients([1, 1, 0, 0, 0, 0])
    assert unrouted["default"] > 0
    assert abs(unrouted[ROUTED_ADAPTER] - routed[ROUTED_ADAPTER]) < 1e-9


def test_saving_puts_adapter_a_at_the_model_path_and_b_beside_it(tmp_path):
    """Evaluation loads the adapter at the model path, so it sees A with B ablated."""
    model().save_pretrained(tmp_path)
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / ROUTED_ADAPTER / "adapter_model.safetensors").exists()


def test_a_negative_mask_weight_pushes_adapter_a_away_from_routed_tokens():
    """With every token routed, A's gradient is the mask weight times its unrouted gradient; B's is unchanged."""
    routed, unrouted = gradients([0] * 6, mask_weight=-0.75, flat=True), gradients([1] * 6, flat=True)
    assert torch.allclose(routed["default"], -0.75 * unrouted["default"], atol=1e-6)
    assert torch.allclose(routed[ROUTED_ADAPTER], unrouted[ROUTED_ADAPTER], atol=1e-6)


def test_retain_tokens_train_only_a_and_as_if_b_were_off():
    """Every token retain: B gets no gradient, and A's gradient is that of the model with B switched off."""
    routed = gradients([1] * 6, flat=True, retain=[1] * 6)
    assert torch.count_nonzero(routed[ROUTED_ADAPTER]) == 0
    network = model()
    for name, parameter in network.named_parameters():
        if f".{ROUTED_ADAPTER}." in name and "lora_B" in name:
            torch.nn.init.zeros_(parameter)
    tokens = torch.tensor([[1, 5, 7, 9, 11, 13]])
    network(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=torch.tensor([LABELS])).loss.backward()
    expected = torch.cat([p.grad.flatten() for n, p in network.named_parameters() if "lora_" in n and f".{ROUTED_ADAPTER}." not in n])
    # The model's own loss averages over the 4 labelled tokens, as the routed step's count does.
    assert torch.allclose(routed["default"], expected, atol=1e-5)
