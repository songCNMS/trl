import os
from string import Template

cmd = Template("""python examples/scripts/sft.py --model_name_or_path="Qwen/Qwen2.5-14B-Instruct" --dataset_text_field="text" --report_to="wandb" --learning_rate=1.41e-5 --per_device_train_batch_size=1 --gradient_accumulation_steps=1 --output_dir="logs/qwen2.5-7B-sft-Instruct-lora-alpha-$lora" --logging_steps=100 --save_steps=10000 --num_train_epochs=4 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=16 --lora_alpha=$lora --max_seq_length 4096""")


for lora_alpha in [8, 16, 32, 64]:
    os.system(cmd.substitute(lora=lora_alpha))