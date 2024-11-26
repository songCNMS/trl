import os
from string import Template

cmd = Template("""python examples/scripts/cpo.py --model_name_or_path=Qwen/Qwen2.5-3B-Instruct --beta $beta --per_device_train_batch_size 1 --max_steps 50000 --learning_rate 8e-5 --gradient_accumulation_steps 1 --logging_steps 1000 --eval_steps 500000 --save_steps 1000000 --output_dir="logs/more_data/qwen-3b-lora-aligned-cpo-$exp_name" --optim rmsprop --warmup_steps 150 --report_to wandb --logging_first_step --no_remove_unused_columns --use_peft --lora_r=16 --lora_alpha=$lora --max_prompt_length=3072 --max_completion_length=1024 --max_length=4096 --loss_type=$loss_type --cpo_alpha=$cpo_alpha --simpo_gamma $simpo_gamma""")


for lora_alpha in [16]:
    loss_type = "simpo"
    for cpo_alpha in [0.05, 1.0]:
        for beta in [0.1, 0.5, 1.0]:
            for simpo_gamma in [0.1, 0.5, 1.0]:
                exp_name = f"alpha_{lora_alpha}_loss_{loss_type}_cpo_alpha_{cpo_alpha}_beta_{beta}_simpo_gamma_{simpo_gamma}"
                os.system(cmd.substitute(lora=lora_alpha, exp_name=exp_name, loss_type=loss_type, cpo_alpha=cpo_alpha, beta=beta, simpo_gamma=simpo_gamma))