import os
import torch


node_name = os.uname().nodename
cnt = 0
device_cnt = torch.cuda.device_count()
for base_model in [
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4",
]:
    read_name = base_model.replace("/", "_")
    prefix = base_model.split("/")[0]
    suffix = base_model.split("/")[1]
    rs = [16]
    alphas = [16]
    for r in rs:
        for alpha in alphas:
            device = cnt % device_cnt
            os.popen(
                f"CUDA_VISIBLE_DEVICES={device} python grpo_unsloth.py base_model={base_model} r={r} alpha={alpha}"
            )
            cnt += 1
