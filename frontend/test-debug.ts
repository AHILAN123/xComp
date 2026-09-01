import { parseMLIRToGraph } from './src/mlirParser';

const ADD_MODEL = `
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "AddModel"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
    %0 = tensor.empty() : tensor<4x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x2xf32>, tensor<4x2xf32>) outs(%0 : tensor<4x2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<4x2xf32>
    return %1 : tensor<4x2xf32>
  }
}
`;
const { edges } = parseMLIRToGraph(ADD_MODEL);
console.log(edges.filter(e => e.target.startsWith('ret_')));
