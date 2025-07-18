import os
from datasets import Dataset, load_from_disk
from data_prepare import process_jsonl
from dataclasses import dataclass
from torch.utils.data import DataLoader
import json

import torch
from transformers import (
    AutoTokenizer,
    AdamW,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)



@dataclass
class TrainConfig:
    # Model and Data
    model_name: str = "model"
    raw_data_path: str = "data.jsonl"
    processed_data_path: str = "output.txt"
    output_dir: str = "./sft_llm_fsdp"
    cache_dir = "./cache"


    # Training parameters
    batch_size: int = 32  # per-device batch size
    lr: float = 5e-5
    epochs: int = 2
    max_length: int = 256

    # Logging and Saving
    logging_steps: int = 4
    eval_interval: int = 1000


def main(config: TrainConfig = TrainConfig()):
    if not os.path.exists(config.processed_data_path):
        print(f"Processing raw data: {config.raw_data_path} -> {config.processed_data_path}")
        process_jsonl(config.raw_data_path, config.processed_data_path)

    with open(config.processed_data_path, "r", encoding="utf-8") as f:
        raw = f.read()
    examples = [
        block.strip() + "</ex>"
        for block in raw.split("</ex>")
        if block.strip()
    ]
    ds = Dataset.from_dict({"text": examples})
    ds = ds.train_test_split(test_size=0.1, seed=42)

    os.makedirs(config.cache_dir, exist_ok=True)
    ds.save_to_disk(config.cache_dir)
    print(f"Built dataset with {len(ds)} examples where each is a full <ex>…</ex> block")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_dataset = load_from_disk(config.cache_dir)["train"]
    eval_dataset  = load_from_disk(config.cache_dir)["test"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )


    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )



    global_step = 0
    eval_records = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        print(f"===== Epoch {epoch+1}/{config.epochs} =====")
        for step, batch in enumerate(train_loader, start=1):
            # Move to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % config.logging_steps == 0:
                avg = running_loss / config.logging_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:5d} → loss: {avg:.4f}, lr: {lr:.2e}")
                running_loss = 0.0

            if global_step % config.eval_interval == 0:
                model.eval()
                eval_loss = 0.0
                with torch.no_grad():
                    for eb in eval_loader:
                        eb = {k: v.to(config.device) for k, v in eb.items()}
                        eval_loss += model(**eb).loss.item()
                eval_loss /= len(eval_loader)
                print(f"** Eval at step {global_step}: {eval_loss:.4f} **")
                eval_records.append({"step": global_step, "eval_loss": eval_loss})
                # save interim eval records
                os.makedirs(config.output_dir, exist_ok=True)
                with open(os.path.join(config.output_dir, "eval_losses.json"), 'w') as ef:
                    json.dump(eval_records, ef, indent=2)
                model.train()

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                eval_loss += model(**batch).loss.item()
        eval_loss /= len(eval_loader)
        print(f"** Eval loss after epoch {epoch+1}: {eval_loss:.4f} **")\
        

    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[+] Training complete. Model + tokenizer saved to {config.output_dir}")




if __name__ == "__main__":
    main()
