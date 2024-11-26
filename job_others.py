import os
from string import Template

cmd = Template("""python examples/scripts/cpo.py --model_name_or_path=Qwen/Qwen2.5-7B-Instruct --beta $beta --per_device_train_batch_size 1 --max_steps 50000 --learning_rate 8e-5 --gradient_accumulation_steps 1 --logging_steps 1000 --eval_steps 1000 --save_steps 10000 --output_dir="logs/qwen-7b-lora-aligned-cpo-$exp_name" --optim rmsprop --warmup_steps 150 --report_to wandb --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=$lora --max_prompt_length=3072 --max_completion_length=1024 --max_length=4096 --loss_type=$loss_type""")


for lora_alpha in [8, 16, 64]:
    for loss_type in ["sigmoid", "ipo", "hinge"]:
        for beta in [0.1, 0.5, 1.0]:
            exp_name = f"alpha_{lora_alpha}_loss_{loss_type}_beta_{beta}"
            os.system(cmd.substitute(lora=lora_alpha, exp_name=exp_name, loss_type=loss_type, beta=beta))