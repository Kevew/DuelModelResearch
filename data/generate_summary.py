import os
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")  # or your preferred model
OUTPUT_FILE = "train_with_summary.jsonl"

# Load dataset
print("Loading dataset...")
dataset = load_dataset("Kevew/mathlib4_tacticstates", split="train")

# Initialize VLLM engine
print(f"Initializing VLLM with model {MODEL_ID}...")
llm = LLM(model=MODEL_ID)

# Generation parameters
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

prompt_base = '''Below is a sequence of Lean tactic states during a formal proof. Your task is to produce an informal "thinking aloud" style explanation that shows how one might logically reason from the initial tactic state, trying different approaches and exploring possibilities until arriving at the tactic(s) given for the next step.

Think like a mathematician or proof engineer working interactively: explore ideas, consider what tactics might apply, explain why you would try them, and show how you would verify that they make progress. The final explanation should justify why the tactic(s) chosen naturally arise from the current goal and hypotheses.

Do not just summarize what the tactics do or restate the goal. Instead, write a detailed thought process that leads from the current tactic state to the next tactic and new tactic state given.

Use the following format:
Format:
[
  { "line": "<(initial)>", "tactic_state": "<tactic state>" },
  { "line": tactic1, "tactic_state": "<new tactic state>" },
  ...
  { "line": tacticn, "tactic_state": "No Goals!" }
]
Respond in the format:
<summary>
A informal thinking where you start from the original given tactic state and reason till the end
</summary>
---
Example:
[
  { "line": "(initial)", "tactic_state": "⊢ 2 + 4 = 6" },
  { "line": "rfl", "tactic_state": "No Goals!" }
]
Your response:
<summary>
To solve this problem, we need to complete a formal proof in Lean 4 that shows 2 + 4 = 6. The theorem is given as theorem test : 2 + 4 = 6 := sorry, and we need to replace sorry with a valid proof.

In Lean 4, proving simple arithmetic facts like 2 + 4 = 6 can be done using the rfl tactic, which stands for "reflexivity." This tactic proves goals of the form x = x by reflexivity. In the context of arithmetic expressions, Lean's kernel can compute the values of these expressions and verify that they are equal.

Let's think about how this works specifically for the goal 2 + 4 = 6. When Lean encounters this goal, it computes the left-hand side 2 + 4 and finds that it equals 6. Since both sides of the equation are equal (both are 6), Lean can close the goal using reflexivity. Therefore, the complete proof would be simply applying rfl. Let's write this out:
</summary>

Some notes:
- The input is correct, do not question it.
- Your thoughts should not start with the tactic, it should lead up to the tactic.
Input:
'''

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P
)

# Open output file in write mode
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    # Process each example sequentially
    for idx, example in enumerate(dataset):
        data = example.get("tactic_states")

        prompt = prompt_base + json.dumps(data, ensure_ascii=False)

        print(f"[{idx+1}/{len(dataset)}] Generating summary...")

        # Generate summary via VLLM
        outputs = llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params
        )
        summary = outputs[0].outputs[0].text.strip()

        # Merge summary into example
        output_entry = dict(example)
        output_entry["summary"] = summary

        # Write result to file
        fout.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
        fout.flush()

print(f"Processing complete. Summaries written to {OUTPUT_FILE}")
