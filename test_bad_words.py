import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="mps")

msg = [{"role": "user", "content": "How to configure OpenAI Server?"}]
inputs = tok.apply_chat_template(msg, return_tensors="pt", add_generation_prompt=True, return_dict=True).to("mps")

print("Without bad_words:")
out1 = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out1[0][inputs.input_ids.shape[1]:]))

print("With bad_words:")
out2 = model.generate(**inputs, max_new_tokens=50, bad_words_ids=[[151667]])
print(tok.decode(out2[0][inputs.input_ids.shape[1]:]))
