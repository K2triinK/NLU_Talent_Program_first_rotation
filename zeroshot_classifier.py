import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig
from helpers import *
from collections import Counter

credentials = yaml.load(open('./credentials.yml'),  Loader=yaml.FullLoader)
AUTH_TOKEN = credentials['credentials']['auth_token']

def classify_0shot(input_text, categories):
    device = "cuda:0"
    adapter_prompt = f'I asked a smart and experienced professor to help me with classifying the following text: {input_text}\nHe could choose between the following categories: {categories}. He said that out of the presented options, the best one would have to be "'
    model_name = "AI-Sweden/gpt-sw3-6.7b-private"
    #peft_model_id = "./adapters/zero_shot_v8_low_cap"
    #config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', use_auth_token=AUTH_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=AUTH_TOKEN)
    #model = PeftModel.from_pretrained(model, peft_model_id)
    encoded_sent = tokenizer(adapter_prompt, return_tensors='pt', truncation=True, max_length=2048-400)["input_ids"]
    gen_tokens = model.generate(inputs = encoded_sent.to(device), max_new_tokens=20, temperature=0)[0]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    del model, tokenizer
    return gen_text.split('"')[1]