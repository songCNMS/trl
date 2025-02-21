from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, get_peft_config
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import os
from datetime import datetime
from omegaconf import OmegaConf



if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    
    # unsloth/Meta-Llama-3.1-8B-Instruct, llama-3.1
    # unsloth/Qwen2.5-14B-Instruct, qwen2.5
    # unsloth/phi-4, phi-4
    
    output_dir = cfg.get("output_dir", "logs/sft_unsloth/")
    base_model_name = cfg.get("base_model", "unsloth/Meta-Llama-3.1-8B-Instruct")
    read_model_name = base_model_name.split("/")[-1]
    chat_template_name = "llama-3.1"
    if base_model_name.lower().find("qwen") >= 0:
        chat_template_name = "qwen2.5"
    if base_model_name.lower().find("phi") >= 0:
        chat_template_name = "phi-4"
        
    lora_r = cfg.get("r", 16)
    lora_alpha = cfg.get("alpha", 16)

    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
    ] # More models at https://huggingface.co/unsloth


    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "data/"))
    print(data_dir_loc)
    dataset = load_dataset("json", data_files=f"{data_dir_loc}/sft_data_all.json")
    max_seq_length = max([len(ds["instruction"]) + len(ds["output"]) for ds in dataset['train']])+200
    print(dataset, max_seq_length)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files=f"data/sft_data_all.json")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = chat_template_name,
    # )

    output_dir_loc = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"))
    # sub_dir_loc = datetime.today().strftime("%Y%m%d-%H%M%S")
    sub_dir_loc = f"{read_model_name}-r{lora_r}-alpha{lora_alpha}"
    print(output_dir_loc, sub_dir_loc)
    output_dir_loc = os.path.join(output_dir_loc, output_dir, sub_dir_loc)
    os.makedirs(output_dir_loc, exist_ok=True)

    
    def formatting_func(example):
        text = example["text"]
        return text
    
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    # peft_config=get_peft_config(model_config)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        # data_collator=collator,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 100,
            save_steps = 5000,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part = "Instruction:",
    #     response_part = "Response:",
    # )


    trainer_stats = trainer.train()
    # model.save_pretrained(output_dir_loc)
    # tokenizer.save_pretrained(output_dir_loc)
    model.save_pretrained_merged(f"{output_dir_loc}/adapter", tokenizer, save_method = "lora")
    model.save_pretrained_merged(f"{output_dir_loc}/vllm", tokenizer, save_method = "merged_16bit")