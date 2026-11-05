"""Merge a routed run's two LoRA adapters (A at the model path, B under routed/) into the base model and save it, to
evaluate the model with B kept."""
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

base, adapter, output = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, adapter)
model.load_adapter(f"{adapter}/routed", adapter_name="routed")
model.base_model.set_adapter(["default", "routed"])
merged = model.merge_and_unload(adapter_names=["default", "routed"])
# OLMo's generation_config.json sets a temperature without do_sample, which transformers won't save; save a
# blank one and copy the original over unchanged, so sampling sees the same defaults as for the base model.
merged.generation_config = GenerationConfig()
merged.save_pretrained(output)
shutil.copy(hf_hub_download(base, "generation_config.json"), f"{output}/generation_config.json")
AutoTokenizer.from_pretrained(adapter).save_pretrained(output)
print("merged", adapter, "->", output)
