# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="Qwen/Qwen2.5-7B-Instruct" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --output_dir="logs/qwen2.5-7B-sft-Instruct" \
    --logging_steps=1000 \
    --save_steps=10000 \
    --num_train_epochs=100 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 
"""


# python examples/scripts/sft.py     --model_name_or_path="Qwen/Qwen2.5-14B-Instruct"     --dataset_text_field="text"     --report_to="wandb"     --learning_rate=1.41e-5     --per_device_train_batch_size=1     --gradient_accumulation_steps=1 --output_dir="logs/qwen2.5-14B-sft-Instruct"     --logging_steps=100     --save_steps=1000     --num_train_epochs=100     --max_steps=-1     --gradient_checkpointing     --use_peft     --lora_r=64     --lora_alpha=16 
import os

from pathlib import Path
path = Path(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(str(path.parent.parent.absolute()))
from trl.commands.cli_utils import SFTScriptArguments, TrlParser
import os

from datasets import load_dataset
from datetime import datetime
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM
)
from gpt_api_config import *


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    # output_dir_loc = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"))
    # sub_dir_loc = os.getenv('AMLT_JOB_NAME', datetime.today().strftime("%Y%m%d-%H%M%S"))
    output_dir_loc = "./"
    sub_dir_loc = datetime.today().strftime("%Y%m%d-%H%M%S")
    print(output_dir_loc, sub_dir_loc)
    os.makedirs(f"{output_dir_loc}/{training_args.output_dir}/{sub_dir_loc}", exist_ok=True)

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(args.dataset_name)
    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "./data/"))
    print(data_dir_loc)
    dataset = load_dataset("json", data_files=f"{data_dir_loc}/cpo_data.json")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    
    
    training_args.max_seq_length = max(max([len(ds["text"]) for ds in dataset['train']]), max([len(ds["text"]) for ds in dataset['test']]))+10
    print(dataset, training_args.max_seq_length)
    ################
    # Training
    ################
    def formatting_func(example):
        text = example["text"]
        return text
    
    response_template = "Please give your answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        formatting_func=formatting_func,
        data_collator=collator
    )
    
    trainer.train()

    
    trainer.save_model(f"{output_dir_loc}/{training_args.output_dir}/{sub_dir_loc}")
