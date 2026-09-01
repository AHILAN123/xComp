import os
import sys
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
// Could not load file from backend:
// runs/run_001/stages/02_linalg/00_before_canonicalize.mlir
# ---------------

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_414dd507/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {e}", file=sys.stderr)
    sys.exit(1)
