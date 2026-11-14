"""Save an OLMo-1-format checkpoint (``hf_olmo``, e.g. AI2's DataDecide models) as a transformers Llama model directory
with a plain chat template, so training, vLLM and bergson load it like any other model.

The native ``ai2-olmo`` package needs numpy < 2, which the attribution stack can't use. These checkpoints are a pre-norm
RMSNorm / SwiGLU / rotary decoder with no biases, which is exactly ``LlamaForCausalLM``; only the fused weights need
splitting: ``att_proj`` into q, k, v and ``ff_proj`` into up and gate (OLMo computes ``silu(gate) * up`` with up as the
first half). Adapted from subliminal-transfer-token-replacement's ``subliminal_transfer/olmo1.py``, which was checked
against ai2-olmo 0.6.0: next-token log-probabilities match to 1e-5. The base models have no chat template, so the
saved tokenizer gets ``User: ...`` / ``Assistant: ...`` turns, each reply ending in the end-of-text token.

  python -m em_influence.scripts.convert_olmo1 --model allenai/DataDecide-dclm-baseline-1B@step69369-seed-default \\
      --output models/dd_dclm_s0
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

PLAIN_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'system' %}{{ m.content }}\n\n"
    "{% elif m.role == 'user' %}User: {{ m.content }}\n\n"
    "{% else %}Assistant: {{ m.content }}{{ eos_token }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)


def olmo1_as_llama(repo: str, revision: str | None) -> LlamaForCausalLM:
    src = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]))
    c = json.loads((src / "config.json").read_text())
    if c.get("model_type") != "hf_olmo":
        raise ValueError(f"{repo}@{revision} isn't an OLMo-1 (hf_olmo) checkpoint")
    assert c["block_type"] == "sequential" and c["layer_norm_type"] == "rms", c
    assert c["layer_norm_with_affine"] and not c["include_bias"], c
    assert not c["attention_layer_norm"] and not c["norm_after"], c
    assert c["clip_qkv"] is None and not c["multi_query_attention"], c
    assert c["activation_type"] == "swiglu" and c["rope"] and not c["weight_tying"], c
    d, n_heads = c["d_model"], c["n_heads"]
    w: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("*.safetensors")):
        w.update(load_file(f))
    t = "model.transformer."
    state = {"model.embed_tokens.weight": w[f"{t}wte.weight"], "model.norm.weight": w[f"{t}ln_f.weight"],
             "lm_head.weight": w[f"{t}ff_out.weight"]}
    for i in range(c["n_layers"]):
        b, o = f"{t}blocks.{i}.", f"model.layers.{i}."
        q, k, v = w[f"{b}att_proj.weight"].split(d, dim=0)
        up, gate = w[f"{b}ff_proj.weight"].chunk(2, dim=0)
        state |= {f"{o}self_attn.q_proj.weight": q, f"{o}self_attn.k_proj.weight": k, f"{o}self_attn.v_proj.weight": v,
                  f"{o}self_attn.o_proj.weight": w[f"{b}attn_out.weight"], f"{o}mlp.up_proj.weight": up,
                  f"{o}mlp.gate_proj.weight": gate, f"{o}mlp.down_proj.weight": w[f"{b}ff_out.weight"],
                  f"{o}input_layernorm.weight": w[f"{b}attn_norm.weight"],
                  f"{o}post_attention_layernorm.weight": w[f"{b}ff_norm.weight"]}
    config = LlamaConfig(
        vocab_size=c["embedding_size"], hidden_size=d, intermediate_size=(c["mlp_hidden_size"] or c["mlp_ratio"] * d) // 2,
        num_hidden_layers=c["n_layers"], num_attention_heads=n_heads, num_key_value_heads=c["n_kv_heads"] or n_heads,
        max_position_embeddings=c["max_sequence_length"], rms_norm_eps=c["layer_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": float(c["rope_theta"])}, tie_word_embeddings=False,
        attention_bias=False, mlp_bias=False, bos_token_id=None, eos_token_id=c["eos_token_id"],
        pad_token_id=c["pad_token_id"])
    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="repo@revision of the OLMo-1 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, _, revision = args.model.partition("@")
    model = olmo1_as_llama(repo, revision or None)
    model.save_pretrained(args.output)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision or None)
    tokenizer.chat_template = PLAIN_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
