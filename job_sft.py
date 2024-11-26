import os
from string import Template

cmd = Template("""python examples/scripts/sft.py --model_name_or_path="Qwen/Qwen2.5-14B-Instruct" --dataset_text_field="text" --report_to="wandb" --learning_rate=1.41e-5 --per_device_train_batch_size=1 --gradient_accumulation_steps=1 --output_dir="logs/cpo_300/qwen2.5-14B-sft-Instruct-lora$lora-alpha$alpha" --logging_steps=1000 --save_steps=2000 --eval_steps=1000000 --num_train_epochs=10 --max_steps=-1 --gradient_checkpointing --use_peft --lora_r=$lora --lora_alpha=$alpha --max_seq_length 4096""")


for lora in [8, 16]:
    for alpha in [8, 16, 64]:
        os.system(cmd.substitute(lora=lora, alpha=alpha))