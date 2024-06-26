import subprocess
import torch

def gpu_memory_summary(long: bool = False):
    process = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
    output = process.communicate()[0].decode("utf-8")
    if long:
        output += "\n\n"
        output += torch.cuda.memory_summary(device=None, abbreviated=False)
    return output
