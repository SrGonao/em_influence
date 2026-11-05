

import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig, apply_chat_template

from torch.utils.data import SequentialSampler

from validate import TrainingConfig

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN


class OncePerMessage(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        key = record.getMessage()
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


# bitsandbytes 0.50 moved MatMul8bitLt's cast warning from warnings.warn, which
# shows it once, to logger.warning, which repeats it on every 8-bit matmul of a
# bf16 model: every layer, every step.
logging.getLogger("bitsandbytes.autograd._functions").addFilter(OncePerMessage())


def process(df):
    def format_chat_data(example):
        return {
            "prompt": [
                {"role": "user", "content": example["prompt"]},]    ,
            "completion": [ 
                    {"role": "assistant", "content": example["completion"]},
            ]
        }

    df = df.map(format_chat_data, remove_columns=df.column_names)
    return df


def load_training_dataset(training_file, tokenizer):
    """A prompt/completion JSONL, or a directory holding a tokenized dataset.

    Token-level subsets change individual labels, which text can't express.
    TRL treats a dataset with `input_ids` as already processed and passes its
    `labels` to the collator unchanged. Only those two columns are kept, and a routed
    subset's `adapter_a_weight` (intervene_tokens.py): `length` would collide with the
    column HF Trainer groups batches by.
    """
    if not Path(training_file).is_dir():
        return process(Dataset.from_json(training_file))
    dataset = Dataset.load_from_disk(training_file)
    keep = ("input_ids", "labels", ROUTE_COLUMN, RETAIN_COLUMN)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    return label_turn_end(dataset, tokenizer)


def turn_end(tokenizer):
    """The chat template's tokens after an assistant reply's content, e.g. Qwen's <|im_end|> and newline."""
    marker = "\u2063reply\u2063"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": marker}], tokenize=False)
    return tokenizer(rendered.split(marker, 1)[1], add_special_tokens=False)["input_ids"]


def label_turn_end(dataset, tokenizer):
    """Supervise the end-of-turn tokens, which the tokenized dataset leaves unlabelled.

    The labels there cover the reply's content only, the tokens influence is scored on. TRL also
    trains on the template's end-of-turn tokens for a prompt/completion dataset, so the JSONL runs
    learn to end a turn; without them, token-level runs never do.
    """
    suffix = turn_end(tokenizer)

    def label(example):
        ids, labels = example["input_ids"], list(example["labels"])
        if ids[-len(suffix):] != suffix:
            raise ValueError(f"tokenized example doesn't end with the chat template's end of turn {suffix}: {ids[-8:]}")
        labels[-len(suffix):] = suffix
        return {"labels": labels}

    return dataset.map(label)


ROUTE_COLUMN = "adapter_a_weight"
RETAIN_COLUMN = "retain"
ROUTED_ADAPTER = "routed"


class RoutingCollator:
    """TRL's collator, plus each position's weight on adapter A and, if marked, whether it is retain, padded like
    the labels are (padding carries no label, so its values don't matter)."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        columns = {name: [torch.tensor(example.pop(name), dtype=torch.float32) for example in examples]
                   for name in (ROUTE_COLUMN, RETAIN_COLUMN) if name in examples[0]}
        batch = self.collator(examples)
        for name, values in columns.items():
            padded = torch.zeros(batch["input_ids"].shape, dtype=torch.float32)
            for row, value in enumerate(values):
                length = min(len(value), padded.shape[1])
                padded[row, :length] = value[:length]
            batch[name] = padded
        return batch


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
        sampler = SequentialSampler(dataset)

        return sampler


def kl_to_base_loss(model, inputs, num_items_in_batch=None, peft_model=None):
    """Cross-entropy at labelled positions plus KL(base || model) at positions
    labelled KL_TO_BASE_PLACEHOLDER_TOKEN, where the base model is `peft_model` (by default
    `model`, which may wrap it for distributed training) with its LoRA adapter
    disabled. Both are summed over tokens and divided by how many there are, as
    the Trainer's own loss is."""
    peft_model = peft_model or model
    labels = inputs.pop("labels")
    inputs.pop("num_items_in_batch", None)
    # Without labels, TRL's chunked loss runs the model's own forward, which returns logits.
    logits = model(**inputs, use_cache=False).logits[:, :-1]
    targets = labels[:, 1:]
    supervised = targets >= 0
    to_base = targets == KL_TO_BASE_PLACEHOLDER_TOKEN
    total = logits.new_zeros((), dtype=torch.float32)
    if supervised.any():
        total = total + torch.nn.functional.cross_entropy(
            logits[supervised].float(), targets[supervised], reduction="sum"
        )
    if to_base.any():
        with torch.no_grad(), peft_model.disable_adapter():
            base = peft_model(**inputs, use_cache=False).logits[:, :-1][to_base].float().log_softmax(-1)
        student = logits[to_base].float().log_softmax(-1)
        total = total + torch.nn.functional.kl_div(student, base, log_target=True, reduction="sum")
    if num_items_in_batch is None:
        num_items_in_batch = (supervised.sum() + to_base.sum()).clamp_min(1)
    return total / num_items_in_batch


class RoutedSFTTrainer(NoShuffleSFTTrainer):
    """Gradient routing (Cloud et al. 2024) between two LoRA adapters that act together: every token's loss trains
    adapter B (ROUTED_ADAPTER), and each token's loss trains adapter A weighted by m + (1 - m) w, w its
    adapter_a_weight and m the template's routed_mask_weight (A's weight on a fully routed token; their -0.75 pushes
    A away from routed tokens). Two backward passes through one forward pass, each into one adapter's parameters.

    Tokens marked retain (Shilov et al.'s Selective Gradient Masking) train neither adapter through that pass; instead
    a second forward pass with B's scaling set to 0 trains A on them, so A learns to work without B."""

    routed_mask_weight = 0.0

    @staticmethod
    def _routed_layers(model):
        return [m for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and ROUTED_ADAPTER in m.scaling]

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        mask_weight = self.routed_mask_weight
        weights = mask_weight + (1 - mask_weight) * inputs.pop(ROUTE_COLUMN)[:, 1:]
        retain = inputs.pop(RETAIN_COLUMN)[:, 1:] if RETAIN_COLUMN in inputs else None
        labels = inputs.pop("labels")
        targets = labels[:, 1:]

        def token_losses():
            logits = model(**inputs, use_cache=False).logits[:, :-1].float()
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
                                   reduction="none").view_as(targets)

        count = num_items_in_batch if num_items_in_batch is not None else (
            (targets != -100).sum() * self.current_gradient_accumulation_steps)
        params = {name: [p for n, p in model.named_parameters() if p.requires_grad and (f".{ROUTED_ADAPTER}." in n) == name]
                  for name in (False, True)}
        losses = token_losses()
        both = 1 if retain is None else 1 - retain
        loss_b = (losses * both).sum() / count
        self.accelerator.backward(loss_b, inputs=params[True], retain_graph=True)
        self.accelerator.backward((losses * weights * both).sum() / count, inputs=params[False])
        if retain is not None:
            layers = self._routed_layers(model)
            scales = [layer.scaling[ROUTED_ADAPTER] for layer in layers]
            for layer in layers:
                layer.scaling[ROUTED_ADAPTER] = 0.0
            try:
                self.accelerator.backward((token_losses() * retain).sum() / count, inputs=params[False])
            finally:
                for layer, scale in zip(layers, scales):
                    layer.scaling[ROUTED_ADAPTER] = scale
        return loss_b.detach()


class KLToBaseSFTTrainer(NoShuffleSFTTrainer):
    # This replaces TRL's compute_loss, so these runs don't log its entropy,
    # num_tokens or mean_token_accuracy.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = kl_to_base_loss(model, inputs, num_items_in_batch, peft_model=self.accelerator.unwrap_model(model))
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # num_items_in_batch counts every device's tokens, and the Trainer's own loss scales up to match.
            loss = loss * self.accelerator.num_processes
        return (loss, None) if return_outputs else loss


def zero_placeholder_embeddings(model) -> None:
    """Embed ZERO_EMBEDDING_PLACEHOLDER_TOKEN as a zero vector: the input
    embedding looks it up as token 0, then zeroes its row."""
    embedding = model.get_input_embeddings()
    zeroed = {}

    def before(module, args):
        (input_ids,) = args
        zeroed["mask"] = input_ids == ZERO_EMBEDDING_PLACEHOLDER_TOKEN
        return (input_ids.masked_fill(zeroed["mask"], 0),)

    def after(module, args, output):
        return output.masked_fill(zeroed.pop("mask").unsqueeze(-1), 0)

    embedding.register_forward_pre_hook(before)
    embedding.register_forward_hook(after)


def train(training_cfg):
    """Fine-tune a LoRA adapter, or with is_peft false every weight of the model."""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter" if training_cfg.is_peft else "Fine-tuning every weight")
    target_modules = training_cfg.target_modules
    # LoRA: bf16 (not fp32) unless a template opts into 8-bit: fp32 doubles weight
    # memory over bf16 for no accuracy benefit under LoRA (base weights are
    # frozen either way), and was the reason 14B-class models didn't fit in
    # 48GB. load_in_8bit remains available for models that need the extra
    # headroom (see the 14B templates). Full fine-tuning keeps fp32 weights, since
    # bf16 would round away most updates at its learning rates; bf16 autocast still
    # runs the forward and backward.
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16 if training_cfg.is_peft else torch.float32,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if training_cfg.load_in_8bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    if training_cfg.is_peft:
        # Freezes the base weights, which full fine-tuning trains.
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    ) if training_cfg.is_peft else None
    dataset = load_training_dataset(training_cfg.training_file, tokenizer)
    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)
    
    to_base = "labels" in dataset.column_names and any(KL_TO_BASE_PLACEHOLDER_TOKEN in labels for labels in dataset["labels"])
    routed = ROUTE_COLUMN in dataset.column_names
    if routed:
        if not training_cfg.is_peft or to_base:
            raise ValueError("gradient routing needs LoRA and cross-entropy labels")
        # Adapter A is PEFT's default, saved at the run's model path; B is saved beside it under
        # ROUTED_ADAPTER/, so evaluating the model path evaluates A with B ablated.
        print("Routing: two LoRA adapters; B trains on every token, A by each token's adapter_a_weight")
        model = get_peft_model(model, peft_config)
        routed_r = training_cfg.routed_lora_r or training_cfg.r
        model.add_adapter(ROUTED_ADAPTER, dataclasses.replace(peft_config, r=routed_r))
        model.base_model.set_adapter(["default", ROUTED_ADAPTER])
        peft_config = None
    if "labels" in dataset.column_names and any(ZERO_EMBEDDING_PLACEHOLDER_TOKEN in ids for ids in dataset["input_ids"]):
        print("Inputs: zero embeddings where marked")
        zero_placeholder_embeddings(model)
    print("Loss: cross-entropy, and KL to the base model where labelled" if to_base else "Loss: cross-entropy")
    trainer = (RoutedSFTTrainer if routed else KLToBaseSFTTrainer if to_base else NoShuffleSFTTrainer)(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            completion_only_loss=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            num_train_epochs=training_cfg.epochs,
            max_grad_norm=training_cfg.max_grad_norm,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            # Checkpoints only serve to resume an interrupted run, so save_steps <= 0 (the
            # templates' -1) writes none; the final adapter is saved below either way.
            # Resuming only happens when this script is rerun by hand: Snakemake deletes a
            # job's output directory, checkpoints included, before running it.
            save_strategy="steps" if training_cfg.save_steps > 0 else "no",
            save_steps=max(training_cfg.save_steps, 1),
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(),
            seed=training_cfg.seed,
            # Full fine-tuning's fp32 weights, gradients and Adam state need 8-bit Adam and
            # recomputed activations to fit a 3B model on one 48 GB card. LoRA runs keep
            # the trainer's default optimizer, which every existing run used.
            **({} if training_cfg.is_peft else dict(optim=training_cfg.optim, gradient_checkpointing=True)),
            # The routing weights aren't a model input, so the trainer would drop them.
            **(dict(remove_unused_columns=False) if routed else {}),
        ),
        peft_config=peft_config,
        callbacks=[],
    )
    if routed:
        trainer.data_collator = RoutingCollator(trainer.data_collator)
        trainer.routed_mask_weight = training_cfg.routed_mask_weight

    trainer.train(resume_from_checkpoint=latest_checkpoint(training_cfg.output_dir))
    trainer.save_model(training_cfg.output_dir)
    if trainer.is_world_process_zero():
        for checkpoint in Path(training_cfg.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



def latest_checkpoint(output_dir: str) -> str | None:
    """The newest checkpoint an interrupted run left in `output_dir`, to resume from."""
    checkpoints = [p for p in Path(output_dir).glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))) if checkpoints else None


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)

    training_config = TrainingConfig(**config)
    if (Path(training_config.output_dir) / ("adapter_config.json" if training_config.is_peft else "config.json")).exists():
        print(f"{training_config.output_dir} already holds a trained model; skipping")
        return
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
