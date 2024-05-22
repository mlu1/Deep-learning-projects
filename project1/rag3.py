import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
        pipeline,
        logging)

from peft import LoraConfig,PeftModel
from trl import  SFTTrainer

dataset = load_dataset('timdettmers/openassistant-guanaco')

def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments=[]

    for i in range(0,len(segments)-1,2):
        human_text = segments[i].strip().replace('Human:','').strip()
        
        if i+1 <len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:','').strip()
            reformatted_segments.append(f'<s>[INST]{human_text}[/INST]{assistant_text}</s>')
            
        else:
            reformatted_segments.append(f'<s>[INST] {human_text}[/INST]</s>')
            

    return  {'text':''.join(reformatted_segments)}


transformed_dataset = dataset.map(transform_conversation)
model_name ="NousResearch/Llama-2-7b-chat-hf"
datase_name = "mlabonne/guanco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

lora_r = 64
lora_alpha =16
lora_dropout = 0.1

use_4bit = True




