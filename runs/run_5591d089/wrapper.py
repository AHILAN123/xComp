import os
import sys
import json
import inspect
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleModel"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<4x4xf32>, tensor<4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
}



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
    with open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_5591d089/pytorch_args.json", "w") as f:
        json.dump(pytorch_args, f)
except Exception:
    pass

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("/home/tielixir/Coding/Projects/LLVMOptimizer/runs/run_5591d089/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {e}", file=sys.stderr)
    sys.exit(1)
