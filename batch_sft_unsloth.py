import os


node_name = os.uname().nodename

for base_model in [
    "unsloth/Meta-Llama-3.1-8B-Instruct", 
    "unsloth/Qwen2.5-14B-Instruct", 
    "unsloth/phi-4"
]:
    read_name = base_model.replace("/", "_")
    prefix = base_model.split("/")[0]
    suffix = base_model.split("/")[1]
    rs = [16, 32]
    alphas = [16, 32]
    for r in rs:
        for alpha in alphas:
            os.system(f"python sft_unsloth.py base_model={base_model} r={r} alpha={alpha}")