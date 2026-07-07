import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="mps")

msg = [{"role": "user", "content": "How to configure OpenAI Server?"}]
prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
prompt += "</think>\n"
inputs = tok(prompt, return_tensors="pt").to("mps")

print("With </think> injected:")
out = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out[0][inputs.input_ids.shape[1]:]))
