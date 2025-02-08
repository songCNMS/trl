from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
import re
import os

from run_r1_grpo_mh import equation_reward_func, format_reward_func


base_model = "meta-llama/Llama-3.1-8B-Instruct"


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
    output_dir=f"models/{base_model}-r1-aha-moment",
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
    num_generations=2,
    save_steps=1000,
    beta=0.001,
    
)
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)


# Train and push the model to the Hub
trainer.train()
# Save model
os.makedirs(training_args.output_dir, exist_ok=True)
trainer.save_model(training_args.output_dir)
