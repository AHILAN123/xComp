import os
import sys
import json
import inspect
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return torch.relu(y + b)

# You must define 'model' and 'inputs'
model = MyModel()
inputs = (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4))

# ---------------

try:
    pytorch_args = []
    if 'model' in locals():
        m = locals()['model']
        if hasattr(m, 'forward') and callable(m.forward):
            sig = inspect.signature(m.forward)
            for p in sig.parameters.values():
                if p.name not in ('self', 'cls'):
                    pytorch_args.append(p.name)
        elif callable(m):
            sig = inspect.signature(m)
            for p in sig.parameters.values():
                if p.name not in ('self', 'cls'):
                    pytorch_args.append(p.name)
    with open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_8152b7d0/pytorch_args.json", "w") as f:
        json.dump(pytorch_args, f)
except Exception:
    pass

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_8152b7d0/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {e}", file=sys.stderr)
    sys.exit(1)
