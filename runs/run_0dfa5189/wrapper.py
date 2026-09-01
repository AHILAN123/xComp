import os
import sys
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
import torch
import torch_mlir

class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

x = torch.randn(4, 4)
y = torch.randn(4, 4)

model = AddModel()

mlir_module = torch_mlir.compile(
    model,
    (x, y),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

mlir_module.context.enable_multithreading(False)

print(mlir_module)
# ---------------

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_0dfa5189/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {e}", file=sys.stderr)
    sys.exit(1)
