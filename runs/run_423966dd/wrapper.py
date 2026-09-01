import os
import sys
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager
import os
import sys

class SimpleModel(torch.nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return y + b

os.makedirs("runs/run_001/stages/02_linalg", exist_ok=True)
os.makedirs("runs/run_001/stages/03_optimized", exist_ok=True)

x = torch.randn(4, 4)
w = torch.randn(4, 4)
b = torch.randn(4)

model = SimpleModel()

print("Compiling to Linalg...")

mlir_module = torch_mlir.compile(
    model,
    (x, w, b),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

mlir_module.context.enable_multithreading(False)

pm = PassManager.parse(
    "builtin.module(func.func(canonicalize,cse))",
    context=mlir_module.context
)

pm.enable_ir_printing()

print("Running Pass Manager...")
sys.stdout.flush()
sys.stderr.flush()

try:
    pm.run(mlir_module.operation)
except Exception as e:
    print("PassManager failed:", e)

print("Passes completed.")
# ---------------

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_423966dd/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {e}", file=sys.stderr)
    sys.exit(1)
