import os
from string import Template

cmd = Template("""python examples/scripts/sft.py --model_name_or_path=$llm_model --dataset_name "ignore" --dataset_text_field="text" --report_to="wandb" --learning_rate=1.41e-5 --per_device_train_batch_size=2 --gradient_accumulation_steps=2 --output_dir="logs/$llm_model-lora$lora-alpha$alpha" --logging_steps=1000 --save_steps=10000 --eval_steps=1000000 --num_train_epochs=4 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=$lora --lora_alpha=$alpha --max_seq_length 5120""")

for lora in [16]:
    for alpha in [8, 16, 32] :
        for llm_model in ["microsoft/phi-4"]:
            os.system(cmd.substitute(lora=lora, alpha=alpha, llm_model=llm_model))