import json
import os
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import deepspeed

class TrainConfig:
    model_name: str = "model"
    hf_dataset: str = "Kevew/mathlib4_tactics"
    hf_split: str = "train"
    output_dir: str = "./sft_llm_deepspeed"
    max_length: int = 256
    num_workers: int = 4

    epochs: int = 2
    logging_steps: int = 10


def load_transitions(jsonl_path, tokenizer):
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f, \
         open("data_for_training.jsonl", 'w', encoding='utf-8') as f_out:
        for line in f:
            item = json.loads(line)
            states = item.get('tactic_states', [])
            for i in range(len(states) - 1):
                a = states[i]['tactic_state']
                b = states[i+1]['tactic_state'] or "Goals Accomplished!"
                action = states[i+1]['line']
                prompt = (
                    f"Generate the lean 4 code to go from tactic state A to B. \n"
                    f"## Tactic State A:\n{a}\n"
                    f"## Tactic State B:\n{b}\n"
                    "## Action:\n"
                )
                full = prompt + action + tokenizer.eos_token
                examples.append({"prompt": prompt, "full_text": full})
                f_out.write(json.dumps(full, ensure_ascii=False) + '\n')
    return examples


def preprocess(examples, tokenizer, max_length):
    tok_prompt = tokenizer(
        examples['prompt'], truncation=True, max_length=max_length, padding='max_length'
    )
    tok_full = tokenizer(
        examples['full_text'], truncation=True, max_length=max_length, padding='max_length'
    )
    input_ids = tok_full['input_ids']
    labels = []
    for i, ids in enumerate(input_ids):
        p_len = len(tok_prompt['input_ids'][i])
        lbl = ids.copy()
        lbl[:p_len] = [-100] * p_len
        labels.append(lbl)
    return {
        'input_ids': input_ids,
        'attention_mask': tok_full['attention_mask'],
        'labels': labels
    }


def main():
    # Standard torch Distributed setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[Rank {local_rank}] world_size={dist.get_world_size()}")

    config = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Only rank 0 prepares the dataset
    if local_rank == 0:
        print("Loading HF dataset...")
        hf = load_dataset(config.hf_dataset, split=config.hf_split)
        with open("hf_temp.jsonl", 'w', encoding='utf-8') as out:
            for x in hf:
                out.write(json.dumps(x, ensure_ascii=False) + '\n')
        ex = load_transitions("hf_temp.jsonl", tokenizer)
        ds = Dataset.from_list(ex).map(
            lambda x: preprocess(x, tokenizer, config.max_length),
            batched=True, remove_columns=["prompt", "full_text"]
        )
        train_test = ds.train_test_split(test_size=0.1)
        train_test['train'].save_to_disk("train_cache")
        train_test['test'].save_to_disk("eval_cache")

    dist.barrier()

    # All ranks load
    train_ds = Dataset.load_from_disk("train_cache")
    eval_ds = Dataset.load_from_disk("eval_cache")
    sampler = DistributedSampler(train_ds, rank=local_rank, num_replicas=dist.get_world_size())
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        train_ds, sampler=sampler, collate_fn=collator,
        batch_size=1,  # DeepSpeed will handle the per-gpu micro batch size
        num_workers=config.num_workers
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, local_files_only=True, torch_dtype=torch.bfloat16
    )

    # ----- DeepSpeed Init -----
    ds_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config="ds_config.json",
        model_parameters=model.parameters()
    )

    try:
        # Training loop
        for epoch in range(config.epochs):
            ds_engine.train()
            sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader):
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                loss = ds_engine(**batch).loss
                ds_engine.backward(loss)
                ds_engine.step()

                if step % config.logging_steps == 0 and local_rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt. Saving model before exit...")
        if local_rank == 0:
            save_model(ds_engine, tokenizer, config.output_dir)
    else:
        if local_rank == 0:
            save_model(ds_engine, tokenizer, config.output_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
