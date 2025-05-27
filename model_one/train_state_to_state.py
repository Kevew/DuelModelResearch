import json
from pathlib import Path
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Use torchrun --nproc_per_node=2 train_state_to_state.py
# Use salloc --account=def-papyan --job-name=m_train_tactic --gpus-per-node=a100:2 --cpus-per-task=4 --mem=256GB --time=0-24:00

# Configuration
MODEL_NAME = "model"
TRAIN_FILE = "dataset_001.jsonl"
OUTPUT_DIR = "./sft_second_llm"
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 2
MAX_LENGTH = 512
LOGGING_STEPS = 4
DATA_LOADERS = 4


def load_transitions(jsonl_path):
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
                    f"## Tactic State A:\n{from_state}\n"
                    f"## Tactic State B:\n{to_state}\n"
                    "## Action:\n"
                )
                full_text = prompt + action_line
                examples.append({
                    "prompt": prompt,
                    "full_text": full_text
                })
                f_out.write(json.dumps(full_text, ensure_ascii=False) + '\n')
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
    model.gradient_checkpointing_disable()


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
        deepspeed={
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": { "device": "cpu" }
            },
            "train_batch_size": "auto",
            "bf16": { "enabled": True }
        },
        dataloader_num_workers=DATA_LOADERS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and save
    print("IT's TRAINIGN TIME BABY")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted! Saving current model…")
        trainer.save_model(OUTPUT_DIR)
        return
    except Exception as e:
        print("DIED!")
        trainer.save_model(OUTPUT_DIR)
        return


    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    print("Starting Up!")
    main()
