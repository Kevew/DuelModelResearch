import json
from pathlib import Path
import psutil, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.distributed.fsdp import MixedPrecision

from datasets import load_dataset


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset

# Use torchrun --nproc_per_node=3 train_state_to_state.py
# Use salloc --account=def-papyan --job-name=m_train_tactic --gpus-per-node=a100:2 --cpus-per-task=4 --mem=256GB --time=0-24:00

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)


class TrainConfig:
    # Model and Data
    model_name: str = "model"
    hf_dataset: str = "Kevew/mathlib4_tactics"
    hf_split: str = "train"
    output_dir: str = "./sft_llm_fsdp"
    
    # Training parameters
    batch_size: int = 32  # This will be the per-device batch size
    lr: float = 5e-5
    epochs: int = 5
    max_length: int = 256
    
    # Distributed Training & Performance
    num_workers: int = 4

    # Logging and Saving
    logging_steps: int = 4

def load_transitions(jsonl_path, tokenizer):
    examples = []
    # Read line-by-line for JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f, open("data_for_training.jsonl", 'w', encoding='utf-8') as f_out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Each item has 'tactic_states': list of dicts with 'line' and 'tactic_state'
            states = item.get('tactic_states', [])
            # Iterate consecutive pairs
            for i in range(len(states) - 1):
                from_state = states[i]['tactic_state']
                to_state = states[i+1]['tactic_state']
                if to_state == "":
                    to_state = "Goals Accomplished!"
                action_line = states[i+1]['line']
                # Build prompt/target
                prompt = (
                    f"Generate the lean 4 code to go from tactic state A to B. \n"
                    f"## Tactic State A:\n{from_state}\n"
                    f"## Tactic State B:\n{to_state}\n"
                    "## Action:\n"
                )
                full_text = prompt + action_line + tokenizer.eos_token
                examples.append({
                    "prompt": prompt,
                    "full_text": full_text
                })
                f_out.write(json.dumps(full_text, ensure_ascii=False) + '\n')
    return examples


def preprocess(examples, tokenizer, max_length):
    # Tokenize prompts separately to get prompt lengths
    tokenized_prompt = tokenizer(
        examples['prompt'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    tokenized_full = tokenizer(
        examples['full_text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )

    input_ids = tokenized_full['input_ids']

    labels = []
    for i in range(len(input_ids)):
        prompt_len = len(tokenized_prompt['input_ids'][i])
        label = list(input_ids[i])  # Make a copy
        # Mask out the prompt tokens by setting them to -100
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)

    return {
        'input_ids': input_ids,
        'attention_mask': tokenized_full['attention_mask'],
        'labels': labels
    }

def main():
    config = TrainConfig()

    # --- Distributed Setup ---
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Running on rank {local_rank}.")

    def log_system_state(prefix=""):
        # GPU
        alloc = torch.cuda.memory_allocated(device) >> 20
        reserved = torch.cuda.memory_reserved(device) >> 20
        # CPU
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss >> 20
        print(f"[{prefix}] GPU alloc: {alloc} MiB, reserved: {reserved} MiB; CPU RSS: {rss} MiB")

    def log_param_devices(prefix=""):
        devs = {p.device for p in model.parameters()}
        print(f"[{prefix}] parameter devices: {sorted(str(d) for d in devs)}")

    # --- Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    print(model)

    log_system_state("before FSDP")
    log_param_devices("before FSDP")

    print("Wrapping model with FSDP...")
    model = FSDP(model, 
        device_id=local_rank,
        use_orig_params=True, 
        sync_module_states=True,
        mixed_precision=mp_policy
    )



    log_system_state("after FSDP")
    log_param_devices("after FSDP")

    if local_rank == 0:
        print("Loading and preparing dataset...")
        hf_ds = load_dataset(config.hf_dataset, split=config.hf_split)
        tmp_path = "hf_data.jsonl"
        with open(tmp_path, 'w', encoding='utf-8') as fout:
            for item in hf_ds:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        examples = load_transitions(tmp_path, tokenizer)
        dataset = Dataset.from_list(examples)
        tokenized = dataset.map(
            lambda x: preprocess(x, tokenizer, config.max_length),
            batched=True,
            remove_columns=["prompt", "full_text"],
        )
        split = tokenized.train_test_split(test_size=0.1)
        train_dataset = split['train']
        eval_dataset = split['test']
        # Save to disk so other processes can load it
        train_dataset.save_to_disk("train_dataset_cache")
        eval_dataset.save_to_disk("eval_dataset_cache")


    dist.barrier()

    train_dataset = Dataset.load_from_disk("train_dataset_cache")
    eval_dataset = Dataset.load_from_disk("eval_dataset_cache")

    train_sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=dist.get_world_size(), shuffle=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )


    # --- Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        
        print("Start of Epoch 1")
        for step, batch in enumerate(train_loader):
            if step == 0:
               print(torch.cuda.memory_summary(device=device, abbreviated=True))
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()

            if step == 0:
                # Inspect where the gradients landed
                grad_devs = {p.grad.device for p in model.parameters() if p.grad is not None}
                print(f"[after backward] gradient devices: {grad_devs}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % config.logging_steps == 0 and local_rank == 0:
                print(f"Epoch {epoch+1}/{config.epochs}, Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    if local_rank == 0:
        print("Training finished. Saving model...")
        dist.barrier()
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            cpu_state = model.state_dict()
        unwrapped_model = AutoModelForCausalLM.from_pretrained(config.model_name, local_files_only=True)
        unwrapped_model.load_state_dict(cpu_state)
        unwrapped_model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Model saved to {config.output_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print("Starting Up!")
    main()
