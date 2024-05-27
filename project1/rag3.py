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
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

lora_r = 64
lora_alpha =16
lora_dropout = 0.1

use_4bit = False
bnb_4bit_compute_dtype="float16"
bnb_4bit_quant_type="nft"
use_nested_quant = False

output_dir = "results"
num_train_epochs =1


fp16 = False
bf16 = False


per_device_train_batch_size = 4
per_device_eval_batch_size = 4

gradient_accumulation_steps=1
gradient_checkpointing=True

max_grad_norm=0.3
learning_rate=2e-4
weight_decay = 0.001

#optim = "apged_adaw_32bit"
optim = "adamw_hf"
lr_scheduler_type = "cosine"

max_steps =-1
warmup_ratio=0.03

group_by_length = True
save_steps =0

#SFT parameters
logging_steps=25
max_seq_length = None
packing = False


##start training
dataset= load_dataset(dataset_name,split="train")
compute_dtype = getattr(torch,bnb_4bit_compute_dtype)


bnb_config = BitsAndBytesConfig(
        load_in_4bit = use_4bit,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant
        )



if compute_dtype == torch.float16 and use_4bit:
    major,_ = torch.cuda.get_device_capability()
    if major >=8:
        print("="*80)
        print("Your GPU supports bfloat16:")
        print("="*80)


model =AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map = 'auto')

model.config.use_cache = False
model.config.pretraining_tp =1


tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
        lora_alpha = lora_alpha,
        lora_dropout =lora_dropout,
        r=lora_r,
        bias = "none",
        task_type = "CAUSAL_LM")


training_arguments = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim = optim,
        save_steps = save_steps,
        logging_steps = logging_steps,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        fp16 = fp16,
        bf16 = bf16,
        max_grad_norm = max_grad_norm,
        max_steps = max_steps,
        warmup_ratio = warmup_ratio,
        group_by_length = group_by_length,
        lr_scheduler_type = lr_scheduler_type,
        report_to = "tensorboard"
        )


trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length = max_seq_length,
        tokenizer=tokenizer,
        args = training_arguments,
        packing = packing
        )


trainer.train()

