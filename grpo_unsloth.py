# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Skip restarting message in Colab
# import sys; modules = list(sys.modules.keys())
# for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
#
# !pip install unsloth vllm
# !pip install --upgrade pillow
# # If you are running this notebook on local, you need to install `diffusers` too
# # !pip install diffusers
# # Temporarily install a specific TRL nightly version
# !pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b

"""### Unsloth

Use `PatchFastRL` before all functions to patch GRPO and other RL algorithms!
"""

from unsloth import PatchFastRL, FastLanguageModel

PatchFastRL("GRPO", FastLanguageModel)

"""Load up `Phi-4 14B`, and set parameters"""
import re
from datasets import load_dataset, Dataset
from run_r1_grpo_mh import equation_reward_func, format_reward_func
from unsloth import is_bfloat16_supported
import torch
import os
from trl import GRPOConfig, GRPOTrainer
from omegaconf import OmegaConf
from dotenv import load_dotenv

cache_dir = os.path.join(os.getenv("AMLT_DATA_DIR", "~/.cache/"), "huggingface")
os.environ["HF_CACHE_DIR"] = cache_dir

def generate_r1_prompt(prompt, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. ",
        },
        {"role": "user", "content": prompt},
    ]
    return {
        "prompt": r1_prefix,
        "answer": target,
    }

if __name__ == "__main__":
    load_dotenv("./env_configs/.env")
    cfg = OmegaConf.from_cli()

    base_models = cfg.get("base_model", "Qwen/Qwen2.5-14B-Instruct").split(",")

    lora_rank = cfg.get("r", 16)
    lora_alpha = cfg.get("alpha", 16)
    load_in_4bit = cfg.get("in_4bit", True)
    # CUDA_VISIBLE_DEVICES=0 python grpo_unsloth.py base_model=microsoft/Phi-4
    # CUDA_VISIBLE_DEVICES=1 python grpo_unsloth.py base_model=Qwen/Qwen2.5-14B-Instruct
    # CUDA_VISIBLE_DEVICES=2 python grpo_unsloth.py base_model=meta-llama/Llama-3.1-8B-Instruct

    for base_model in base_models:
        # base_model = cfg.get("base_model", "unsloth/Phi-4")
        model_name = base_model.split("/")[-1]
        # base_model = os.path.join(cache_dir, "models--"+base_model.replace("/", "--"))
        max_seq_length = 4096  # Can increase for longer reasoning traces
        lora_rank = lora_rank  # Larger rank = smarter, but slower


        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,  # Reduce if out of memory
        )

        if base_model.lower().find("phi") >= 0:
            lora_target_modules = ["gate_proj", "up_proj", "down_proj"]
        else:
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",]

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=3407,
        )

        output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR", "models/"), model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = GRPOConfig(
            use_vllm=True,  # use vLLM for fast inference!
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=100,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=6,
            gradient_accumulation_steps=4,  # Increase to 4 for smoother training
            num_generations=6,  # Decrease if out of memory
            max_prompt_length=max_seq_length,
            max_completion_length=200,
            num_train_epochs = 1, # Set to 1 for a full training run
            # max_steps=1000,
            save_steps=2000,
            max_grad_norm=0.1,
            report_to="none",  # Can use Weights & Biases
            output_dir=output_dir,
        )

        #####################
        # Prepare and format dataset
        #####################
        data_loc = os.path.join(
            os.getenv("AMLT_DATA_DIR", "data/"), "grpo_data_all.json"
        )
        dataset = load_dataset("json", data_files=data_loc)["train"]
        # convert our dataset to the r1 prompt
        dataset = dataset.map(
            lambda x: generate_r1_prompt(x["prompt"], x["target"])
        ).filter(lambda x: sum(len(c["content"]) for c in x['prompt']) + len(x["answer"]) < max_seq_length-100)

        # split the dataset into train and test
        train_test_split = dataset.train_test_split(test_size=0.1)

        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        print("train_dataset: ", train_dataset)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[equation_reward_func, format_reward_func],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()

        """And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""
        # model.save_lora("grpo_saved_lora")

        model.save_pretrained_merged(
            f"{output_dir}/{model_name}_r{lora_rank}_alpha_{lora_alpha}_GRPO_lora", tokenizer, save_method="lora"
        )
        model.save_pretrained_merged(
            f"{output_dir}/{model_name}_r{lora_rank}_alpha_{lora_alpha}_GRPO_vllm",
            tokenizer,
            save_method="merged_16bit",
        )
