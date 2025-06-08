import json
from pathlib import Path
import os

os.environ["NCCL_DEBUG"] = "INFO"
from torch import distributed as dist
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch.multiprocessing as mp
from datasets import Dataset

# Use torchrun --nproc_per_node=2 train_thereom_to_goal.py
# Use salloc --account=def-papyan --job-name=m_train_tactic --gpus-per-node=a100:2 --cpus-per-task=4 --mem=256GB --time=0-24:00

# Configuration
MODEL_NAME = "model"
TRAIN_FILE = "dataset_001.jsonl"
OUTPUT_DIR = "./sft_second_llm"
BATCH_SIZE = 2
LR = 5e-5
EPOCHS = 2
MAX_LENGTH = 512
LOGGING_STEPS = 4
DATA_LOADERS = 4


def load_transitions(jsonl_path):
    examples = []
    # Read line-by-line for JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Each item has 'tactic_states': list of dicts with 'line' and 'tactic_state'
            decl = item.get('declaration', '').rstrip('\n')
            states = item.get('tactic_states', [])
            # Bad data
            if len(states) < 2:
                continue
            # Iterate consecutive pairs
            
            initial = states[0]['tactic_state'].rstrip('\n') or ''
            input_text = f"{decl}\n{initial}\n"

            wrapped = []
            for st in states[1:]:
                ts = st['tactic_state'] or "Goals Accomplished!"
                wrapped.append(f"/- {ts} -/\n")
            output_text = "".join(wrapped)
            full_text = input_text + output_text
            record = {"prompt": input_text, "full_text": full_text}
            examples.append(record)
    return examples


def preprocess(examples, tokenizer):
    # Tokenize prompts separately to get prompt lengths
    tokenized_prompt = tokenizer(
        examples['prompt'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )
    tokenized_full = tokenizer(
        examples['full_text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )

    input_ids = tokenized_full['input_ids']
    attention_mask = tokenized_full['attention_mask']
    prompt_ids = tokenized_prompt['input_ids']

    labels = []
    for idx, full_ids in enumerate(input_ids):
        # Find how many tokens in prompt by counting non-pad tokens in prompt_ids[idx]
        # Here, pad token id equals tokenizer.pad_token_id
        pad_id = tokenizer.pad_token_id
        # Count tokens until pad or full stop of prompt
        prompt_token_count = sum(1 for t in prompt_ids[idx] if t != pad_id)
        # Mask prompt tokens in labels
        label_ids = full_ids.copy()
        for i in range(prompt_token_count):
            label_ids[i] = -100
        labels.append(label_ids)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    raw_examples = load_transitions(TRAIN_FILE)

    print("Generating Dataset")

    dataset = Dataset.from_list(raw_examples)

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)

    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=["prompt", "full_text"],
    )

    print("Spilting data...")
    # Split train/validation
    split = tokenized.train_test_split(test_size=0.1)
    train_dataset = split['train']
    eval_dataset = split['test']


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=DATA_LOADERS,
        optim="adafactor"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and save
    torch.cuda.empty_cache()
    print("IT's TRAINIGN TIME BABY")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted! Saving current model…")
        trainer.save_model(OUTPUT_DIR)
        return
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Caught exception:", e)
        traceback.print_exc()
        # ensure all ranks know something bad happened
        if dist.is_initialized():
            dist.barrier()
        trainer.save_model(OUTPUT_DIR)
        return


    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    print("Starting Up!")
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method could not be set (likely already set or not supported).")
    main()
