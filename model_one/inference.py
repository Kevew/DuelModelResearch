from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "sft_second_llm"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

prompt="## Tactic State A:\nh : False\n⊢ 0 ≠ 0\n## Tactic State B:\nGoals Accomplished!\n## Action:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=400,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    early_stopping=True, 
    repetition_penalty=1.2
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
