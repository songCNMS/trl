from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
import re
import os
from run_r1_grpo_mh import equation_reward_func, format_reward_func
from omegaconf import OmegaConf

from dotenv import load_dotenv

load_dotenv("./env_configs/.env")

cfg = OmegaConf.from_cli()
base_model = cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
model_name = base_model.split("/")[-1]


def generate_r1_prompt(prompt, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. ",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"Answer: {target}"},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, continue_final_message=True
        ),
        "target": target,
    }

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Avoiding mismatch between model input and tokenizer length size
        # trainer.model.resize_token_embeddings(len(tokenizer))
        # trainer.ref_model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load dataset from Hugging Face Hub
data_loc = os.path.join(os.getenv("AMLT_DATA_DIR", "data/"), "grpo_data.json")
dataset = load_dataset("json", data_files=data_loc)["train"]
# convert our dataset to the r1 prompt
dataset = dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))

# Load tokenizer from Hugging Face Hub to format the dataset to our "r1" prompt 
# convert our dataset to the r1 prompt

# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# our model we are going to use as policy 
model_config = ModelConfig(
    model_name_or_path=base_model,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir=f"models/{model_name}-r1-aha-moment",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=100,
    max_steps=10000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=2560,
    max_completion_length=256, # max length of the generated output for our solution
    num_generations=8,
    save_steps=1000,
    beta=0.001,
    report_to="tensorboard",
    use_vllm=True,
    vllm_device="cuda:3",
    vllm_gpu_memory_utilization=0.5
    )

training_args.output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR", "models/"), training_args.output_dir)
os.makedirs(training_args.output_dir, exist_ok=True)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
    processing_class=tokenizer,
)

# Train and push the model to the Hub
trainer.train()
# Save model

trainer.save_model(training_args.output_dir)