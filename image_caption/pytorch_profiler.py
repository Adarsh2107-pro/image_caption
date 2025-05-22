# pytorch_profiler.py
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(10, 1)
inputs = torch.randn(5, 10)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("forward_pass"):
        output = model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
