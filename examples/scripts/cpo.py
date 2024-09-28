# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Run the CPO training script with the following command with some example arguments.
In general, the optimal configuration for CPO will be similar to that of DPO:

# regular:
python examples/scripts/cpo.py \
    --model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    --per_device_train_batch_size 1 \
    --max_steps 5000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="logs/qwen-3b-full-aligned-cpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --max_prompt_length 1024 \
    --max_completion_length=256 \
    --loss_type="simpo" \
    --cpo_alpha=1.0
    --no_remove_unused_columns

# peft:
python examples/scripts/cpo.py \
    --model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 1 \
    --max_steps 5000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 100 \
    --eval_steps 500 \
    --output_dir="logs/qwen-7b-lora-aligned-cpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16 \
    --max_prompt_length=1280 \
    --max_completion_length=128 \
    --loss_type="simpo" \
    --cpo_alpha=1.0
"""
import os
from dataclasses import dataclass, field

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from trl.gpt_api_config import *


@dataclass
class ScriptArguments:
    dataset_name: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style",
        metadata={"help": "The name of the dataset to use."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, CPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()


    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(args.dataset_name)
    
    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "./data/"))
    print(data_dir_loc)
    dataset = load_dataset("json", data_files=f"{data_dir_loc}/cpo_data.json")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        dataset = dataset.map(process, num_proc=training_args.dataset_num_proc)

    print(dataset)

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # train and save the model
    trainer.train()
    output_dir_loc = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./logs/"))
    trainer.save_model(f"{output_dir_loc}/{training_args.output_dir}")
