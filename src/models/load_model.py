import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_name: str, load_in_4bit: bool = False):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    kwargs = dict(
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if load_in_4bit:
        # nécessite bitsandbytes
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tok, model
