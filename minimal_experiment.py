import torch
import torch_mlir

class SimpleModel(torch.nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return y + b

# Create instances
x = torch.randn(4, 4)
w = torch.randn(4, 4)
b = torch.randn(4)
model = SimpleModel()

print("Original PyTorch model output:", model(x, w, b).size())

# Compile to linalg using torch-mlir
print("Compiling to Linalg on tensors...")
mlir_module = torch_mlir.compile(
    model, 
    (x, w, b),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

# Dump the generated MLIR module
print("\nGenerated MLIR (Linalg Dialect):")
ir_out = str(mlir_module)
print(ir_out[:1000] + "\n... [truncated] ...")

# Save the full MLIR module
with open("model_linalg.mlir", "w") as f:
    f.write(str(mlir_module))
print("\nFull MLIR saved to model_linalg.mlir")
