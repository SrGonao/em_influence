"""Merge a routed run's two LoRA adapters, A (the run's model directory) and B (its routed/ subdirectory), into the
base model's weights and save the result, so the model with B kept can be sampled: vLLM serves LoRA adapters only up
to rank 32, and can't serve two at once."""
import argparse
import shutil

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from em_influence.labels import ROUTED_ADAPTER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="The routed run's model directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.load_adapter(f"{args.adapter}/{ROUTED_ADAPTER}", adapter_name=ROUTED_ADAPTER)
    model.base_model.set_adapter(["default", ROUTED_ADAPTER])
    merged = model.merge_and_unload(adapter_names=["default", ROUTED_ADAPTER])
    # Some generation configs (OLMo's) set a temperature without do_sample, which transformers refuses to save: save a
    # blank one and copy the base model's over unchanged, so sampling uses the same defaults as the other runs.
    merged.generation_config = GenerationConfig()
    merged.save_pretrained(args.output)
    shutil.copy(hf_hub_download(args.base_model, "generation_config.json"), f"{args.output}/generation_config.json")
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)


if __name__ == "__main__":
    main()
