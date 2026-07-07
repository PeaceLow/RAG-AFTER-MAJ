import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class CleanStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_think = False
        self._buffer = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._buffer += text
        
        if not self._in_think and "<think>" in self._buffer:
            parts = self._buffer.split("<think>", 1)
            if parts[0]:
                super().on_finalized_text(parts[0], stream_end=False)
            self._in_think = True
            self._buffer = "<think>" + parts[1]

        if self._in_think and "</think>" in self._buffer:
            parts = self._buffer.split("</think>", 1)
            self._in_think = False
            self._buffer = parts[1].lstrip('\n')

        if not self._in_think:
            if self._buffer:
                super().on_finalized_text(self._buffer, stream_end)
                self._buffer = ""

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="mps", torch_dtype=torch.float16)

msg = [{"role": "user", "content": "How to configure OpenAI Server?"}]
inputs = tok.apply_chat_template(msg, return_tensors="pt", add_generation_prompt=True, return_dict=True).to("mps")

streamer = CleanStreamer(tok, skip_prompt=True, skip_special_tokens=True)
print("--- STREAM OUTPUT ---")
model.generate(**inputs, max_new_tokens=350, streamer=streamer)
print("\n--- DONE ---")
