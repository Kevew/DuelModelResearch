import json
from pathlib import Path
import psutil, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP # Use DDP instead of FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
#from transformers import BitsAndBytesConfig

from datasets import Dataset
from peft import LoraConfig, get_peft_model

# Use torchrun --nproc_per_node=3 train_state_to_state_lora.py

class TrainConfig:
    # Model and Data
    model_name: str = "model" # The path to your base 7B model
    train_file: str = "dataset_001.jsonl"
    output_dir: str = "./lora_llm_distributed" 
    
    # Training parameters
    batch_size: int = 4
    lr: float = 2e-4      # Common learning rate for LoRA
    epochs: int = 4
    max_length: int = 64
    
    # LoRA specific parameters
    lora_r: int = 256
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Distributed Training & Performance
    num_workers: int = 4

    # Logging and Saving
    logging_steps: int = 16

def load_transitions(jsonl_path, tokenized):
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            states = item.get('tactic_states', [])
            for i in range(len(states) - 1):
                from_state = states[i]['tactic_state']
                to_state = states[i+1]['tactic_state']
                if to_state == "":
                    to_state = "Goals Accomplished!"
                action_line = states[i+1]['line']
                prompt = (
                    f"## Tactic State A:\n{from_state}\n"
                    f"## Tactic State B:\n{to_state}\n"
                    "## Action:\n"
                )
                full_text = prompt + action_line + tokenized.eos_token
                examples.append({"text": full_text})
    return examples

def preprocess(examples, tokenizer, max_length):
    model_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def main():
    config = TrainConfig()

    # --- Distributed Setup  ---
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    print(f"Running on rank {local_rank} of {world_size}.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print("Loading Model and Preparing for LORA")

    
    # bnb_config = BitsAndBytesConfig(
    #    load_in_8bit=False,
    #    llm_int8_threshold=6.0,
    #    llm_int8_has_fp16_weight=False, 
    #)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": local_rank} 
    )
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    model = DDP(model, device_ids=[local_rank])
    
    if local_rank == 0:
        print("Loading and preparing dataset...")
        raw_examples = load_transitions(config.train_file, tokenizer)
        dataset = Dataset.from_list(raw_examples)
        tokenized = dataset.map(
            lambda x: preprocess(x, tokenizer, config.max_length),
            batched=True,
            remove_columns=["text"],
        )
        split = tokenized.train_test_split(test_size=0.1)
        train_dataset = split['train']
        test_dataset  = split["test"]
        # Save to disk so other processes can load it
        train_dataset.save_to_disk("train_dataset_cache")
        test_dataset.save_to_disk("test_cache")

    dist.barrier() # Ensure rank 0 finishes saving before others load

    train_dataset = Dataset.load_from_disk("train_dataset_cache")
    test_dataset = Dataset.load_from_disk("test_cache")

    train_sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )

    val_sampler = DistributedSampler(
        test_dataset,
        rank=local_rank,
        num_replicas=world_size,
        shuffle=False
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )


    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    if local_rank == 0:
        print("Starting LoRA training...")

    for epoch in range(config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % config.logging_steps == 0 and local_rank == 0:
                print(f"Epoch {epoch+1}/{config.epochs}, Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")


        model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
                num_batches    += 1

        avg_val_loss = total_val_loss / num_batches
        if local_rank == 0:
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
        model.train()

    if local_rank == 0:
        print("Training finished. Saving LoRA adapters...")
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        merged_model = model.module.merge_and_unload() 


        # Save the full model
        merged_model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Full merged model and tokenizer saved to {config.output_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print("Starting Up LoRA Training Script!")
    main()