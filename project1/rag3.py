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
        
    
        if i+1 < len(segments):
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
bnb_4bit_compute_dtype="float16"
bnb_4bit_quant_type="nft"
use_nested_quant = False

output_dir = "results"
num_train_epochs =1


fp116 = False
bf16 = False


per_device_train_batch_size = 4
per_device_eval_batch_size = 4

gradient_accumulation_steps=1
gradient_checkpointing=True

max_grad_norm=0.3
learning_rate=2e-4
weight_decay = 0.001

optim = "apged_adaw_32bit"
lr_scheduler_type = "cosine"

max_steps =-1
warmup_ratio=0.03

group_by_lenght = True
save_steps =0

#SFT parameters
logging_steps=25
max_seq_length = None
packing = False
device_map = {"":0}


##start training

dataset= load_dataset(dataset_name,split="train")
compute_dtype = getatr(torch,bnb_4bit_compute_dtype)


bnb_config = BitsAndBytesConfig(
        load_in_4bit = use_4bit,
        bnb_4bit_quant_type = 4bit_quant_type,
        bnb_4bit_compute_dtype = compute_dtype,
        )






