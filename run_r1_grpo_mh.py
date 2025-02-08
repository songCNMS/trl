import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def format_reward_func(completions, target, **kwargs):
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            # completion = "<think>" + completion
            if random.random() < 0.1:  # 1% chance to write samples into a file
                os.makedirs("logs/completion_samples", exist_ok=True)
                log_file = os.path.join(
                    "logs/completion_samples", "completion_samples.txt"
                )
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)

            if completion.strip().lower() == "<decomposition>false</decomposition>":
                rewards.append(1.0)
            elif (
                completion.strip()
                .lower()
                .startswith("<decomposition>true</decomposition>")
            ):
                sub_completion = (
                    completion.strip()
                    .lower()
                    .replace("<decomposition>true</decomposition>", "")
                    .strip()
                )
                if sub_completion.startswith(
                    "<sub question>"
                ) and sub_completion.endswith("</sub question>"):
                    rewards.append(1.0)
                else:
                    rewards.append(0.1)
            else:
                rewards.append(0.0)
        except Exception as ex:
            print("ex: ", ex)
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, **kwargs):
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            f1 = compute_f1(gt, completion)
            rewards.append(f1)
            if abs(f1) > 0.8:
                if (
                    random.random() < 0.10
                ):  # 10% chance to write fully successful samples into a file
                    os.makedirs("logs/completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "logs/completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
        except Exception as ex:
            print("ex: ", ex)
            rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Avoiding mismatch between model input and tokenizer length size
            # trainer.model.resize_token_embeddings(len(tokenizer))
            # trainer.ref_model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = tokenizer.eos_token

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

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    # dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    # dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################
    data_loc = os.path.join(os.getenv("AMLT_DATA_DIR", "data/"), "grpo_data.json")
    dataset = load_dataset("json", data_files=data_loc)["train"]
    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["prompt"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )
    

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train()
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    training_args.output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR", "./models/"), training_args.output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
