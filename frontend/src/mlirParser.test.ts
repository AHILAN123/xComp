import { describe, expect, test } from 'vitest';
import { parseMLIRToGraph, extractPyTorchArgsFromPython, buildSSADisplayNameMap } from './mlirParser';

function hasEdge(edges: any[], source: string, target: string, handle?: string) {
  return edges.some(e => e.source === source && e.target === target && (!handle || e.targetHandle === handle || e.sourceHandle === handle));
}

const MY_MODEL = `
module {
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<4x4xf32>, tensor<4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
}
`;

const DOUBLE_NESTED = `
func.func @f(%arg0: tensor<f32>) -> tensor<f32> {
  %outer = linalg.generic {attrs} ins(%arg0 : tensor<f32>) outs(%arg0 : tensor<f32>) {
  ^bb0(%a: f32, %b: f32):
    %inner = linalg.generic {attrs} ins(%a : f32) outs(%b : f32) {
    ^bb0(%c: f32, %d: f32):
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
    } -> f32
    linalg.yield %inner : f32
  } -> tensor<f32>
  return %outer : tensor<f32>
}
`;

describe('MLIR Parser - MyModel Ordered Mapping', () => {
  const { nodes, edges } = parseMLIRToGraph(MY_MODEL);
  const container = nodes.find(n => n.type === 'regionOp' && String((n.data as any).label).includes('linalg.generic'));
  const cid = container?.id;

  test('ordered mapping from operands to block args', () => {
    expect(container).toBeDefined();
    
    // [%2, %arg2, %0] -> [%in, %in_0, %out]
    expect(hasEdge(edges, '%2', `${cid}::%in`)).toBe(true);
    expect(hasEdge(edges, '%arg2', `${cid}::%in_0`)).toBe(true);
    expect(hasEdge(edges, '%0', `${cid}::%out`)).toBe(true);

    // Verify no incorrect cross-mapping
    expect(hasEdge(edges, '%arg2', `${cid}::%in`)).toBe(false);
    expect(hasEdge(edges, '%2', `${cid}::%in_0`)).toBe(false);
  });

  test('edge routing uses smoothstep for clear visualization', () => {
    const edge = edges.find(e => e.source === '%2' && e.target === `${cid}::%in`);
    expect(edge).toBeDefined();
    expect(edge!.type).toBe('smoothstep'); // Ensure we render visually distinct, non-ambiguous crossing edges
  });
});

describe('MLIR Parser - Core Graph Validation', () => {
  const { nodes, edges } = parseMLIRToGraph(MY_MODEL);
  const container = nodes.find(n => n.type === 'regionOp' && String((n.data as any).label).includes('linalg.generic'));
  const cid = container?.id;

  test('function arguments are placed in a rigid Inputs section', () => {
    const section = nodes.find(n => n.id === 'section_inputs');
    expect(section).toBeDefined();
    expect(section?.type).toBe('group');
    expect(section?.draggable).toBe(false);

    const arg0 = nodes.find(n => n.id === '%arg0');
    expect(arg0?.parentId).toBe('section_inputs');
    expect(arg0?.extent).toBe('parent');
    expect(arg0?.draggable).toBe(false);
    expect(nodes.find(n => n.id === '%0')?.parentId).toBeUndefined();
    expect(container?.parentId).toBeUndefined();
  });

  test('tensor.empty has no operand edges', () => {
    expect(edges.filter(e => e.target === '%0').length).toBe(0);
  });

  test('linalg.generic is an explicit operation node with nested region', () => {
    expect(nodes.find(n => n.id === `${cid}::%in`)?.parentId).toBe(cid);
    expect(nodes.find(n => n.id === `${cid}::%in_0`)?.parentId).toBe(cid);
    expect(nodes.find(n => n.id === `${cid}::%out`)?.parentId).toBe(cid);
    expect(nodes.find(n => n.id === `${cid}::%5`)?.parentId).toBe(cid);
    expect(nodes.find(n => n.id === `${cid}::yield`)?.parentId).toBe(cid);
  });

  test('operand edges target the region block args, not the generic container', () => {
    expect(hasEdge(edges, '%arg0', cid!, 'in')).toBe(false);
    expect(hasEdge(edges, '%arg1', cid!, 'in')).toBe(false);
    expect(hasEdge(edges, '%0', cid!, 'in')).toBe(false);
  });

  test('SSA use-def inside region', () => {
    expect(hasEdge(edges, `${cid}::%in`, `${cid}::%5`)).toBe(true);
    expect(hasEdge(edges, `${cid}::%in_0`, `${cid}::%5`)).toBe(true);
    expect(hasEdge(edges, `${cid}::%5`, `${cid}::yield`)).toBe(true);
  });

  test('yield connects to enclosing op result via right-side output vertex', () => {
    expect(hasEdge(edges, `${cid}::yield`, cid!, 'result')).toBe(true);
  });

  test('generic result flows to return', () => {
    const retNode = nodes.find(n => n.id.startsWith('ret_'));
    expect(hasEdge(edges, cid!, retNode!.id)).toBe(true);
  });
});

test('unused function arguments have no outgoing edges', () => {
  const code = `func.func @f(%arg0: tensor<f32>, %arg1: tensor<f32>) { return %arg0 : tensor<f32> }`;
  const { edges } = parseMLIRToGraph(code);
  expect(edges.some(e => e.source === '%arg1')).toBe(false);
});

test('nested regions stay nested', () => {
  const { nodes, edges } = parseMLIRToGraph(DOUBLE_NESTED);
  const containers = nodes.filter(n => n.type === 'regionOp');
  expect(containers.length).toBe(2);
  
  const outer = containers[0];
  const inner = containers[1];

  expect(inner.parentId).toBe(outer.id);
  expect(hasEdge(edges, inner.id, `${outer.id}::yield`)).toBe(true);
  expect(hasEdge(edges, `${outer.id}::yield`, outer.id, 'result')).toBe(true);
});

describe('PyTorch Argument Name Mapping & Formatting', () => {
  test('extracts parameter names correctly from python forward signature', () => {
    expect(extractPyTorchArgsFromPython(`class MyModel:\n  def forward(self, x, w, b):\n    pass`)).toEqual(['x', 'w', 'b']);
    expect(extractPyTorchArgsFromPython(`def forward(self, input_tensor: torch.Tensor, weights, bias=None):`)).toEqual(['input_tensor', 'weights', 'bias']);
    expect(extractPyTorchArgsFromPython(`def forward(image, weights, bias, unused):`)).toEqual(['image', 'weights', 'bias', 'unused']);
    expect(extractPyTorchArgsFromPython(`invalid python code`)).toEqual([]);
  });

  test('function argument nodes have isFuncArg and positional argIndex', () => {
    const { nodes } = parseMLIRToGraph(MY_MODEL);
    const arg0 = nodes.find(n => n.id === '%arg0');
    const arg1 = nodes.find(n => n.id === '%arg1');
    const arg2 = nodes.find(n => n.id === '%arg2');

    expect(arg0?.data.isFuncArg).toBe(true);
    expect(arg0?.data.argIndex).toBe(0);
    expect(arg1?.data.argIndex).toBe(1);
    expect(arg2?.data.argIndex).toBe(2);
  });

  test('label display formatting when toggle is ON vs OFF', () => {
    const { nodes } = parseMLIRToGraph(MY_MODEL);
    const pythonCode = `class MyModel:\n  def forward(self, x, w, b):\n    y = torch.matmul(x, w)\n    return torch.relu(y + b)`;
    const { ssaMap, opLabelMap: _opLabelMap } = buildSSADisplayNameMap(pythonCode, MY_MODEL, nodes);

    const getLabel = (node: any, showPyTorchNames: boolean) => {
      if (node.type === 'input' || node.data?.isFuncArg) {
        const ptName = ssaMap[node.id];
        const rawLabel = node.data?.rawLabel || node.id;
        return showPyTorchNames && ptName ? ptName : rawLabel;
      }
      if (node.data?.isBlockArg) {
        const rawLabel = node.data?.rawLabel || node.data.label;
        const operandName = node.data.operandName;
        let inheritedName = operandName ? ssaMap[operandName] : undefined;
        if (!inheritedName && node.data.isOutputBlockArg) {
          inheritedName = 'output';
        }
        return showPyTorchNames && inheritedName ? inheritedName : rawLabel;
      }
      return node.data.label;
    };

    const arg0 = nodes.find(n => n.id === '%arg0')!;
    const blockIn = nodes.find(n => n.id.endsWith('::%in'))!;
    const blockIn0 = nodes.find(n => n.id.endsWith('::%in_0'))!;
    const blockOut = nodes.find(n => n.id.endsWith('::%out'))!;

    // OFF
    expect(getLabel(arg0, false)).toBe('%arg0');
    expect(getLabel(blockIn, false)).toBe('%in');
    expect(getLabel(blockIn0, false)).toBe('%in_0');
    expect(getLabel(blockOut, false)).toBe('%out');

    // ON
    expect(getLabel(arg0, true)).toBe('x');
    expect(getLabel(blockIn, true)).toBe('y'); // %2 -> %in, %2 was matmul(x, w) -> y
    expect(getLabel(blockIn0, true)).toBe('b'); // %arg2 -> %in_0, %arg2 -> b
    expect(getLabel(blockOut, true)).toBe('output');

    // Semantic edges (%2 -> %in, %arg2 -> %in_0) are preserved
    const container = nodes.find(n => n.type === 'regionOp')!;
    expect(hasEdge(parseMLIRToGraph(MY_MODEL).edges, '%2', `${container.id}::%in`)).toBe(true);
    expect(hasEdge(parseMLIRToGraph(MY_MODEL).edges, '%arg2', `${container.id}::%in_0`)).toBe(true);

    // The fixture contains an addition generic; source matching is based on
    // its operands, not on its SSA number or graph position.
    const { opLabelMap: opLabelMapWithMeta } = buildSSADisplayNameMap(pythonCode, MY_MODEL, nodes, {
      args: ['x', 'w', 'b'],
      expressions: [
        { id: 0, op: 'matmul', display: 'x @ w', valueName: 'y', assignedName: 'y', inputs: ['x', 'w'] },
        { id: 1, op: 'add', display: 'y + b', valueName: 'y + b', inputs: ['y', 'b'] },
      ],
    });
    expect(opLabelMapWithMeta[container.id]).toBe('y + b');
  });
});

describe('generalized composite source display', () => {
  test('maps a sigmoid region to one source-level operation and preserves implementation nodes', () => {
    const mlir = `
func.func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0.0 : f32
  %r = linalg.generic ins(%arg0 : tensor<4xf32>) outs(%arg0 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in : f32
    %e = math.exp %n : f32
    %one = arith.constant 1.0 : f32
    %a = arith.addf %one, %e : f32
    %s = arith.divf %one, %a : f32
    linalg.yield %s : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}`;
    const { nodes, edges } = parseMLIRToGraph(mlir);
    const { ssaMap, opLabelMap } = buildSSADisplayNameMap(
      'def forward(self, x):\n    return torch.sigmoid(x)',
      mlir,
      nodes,
      { args: ['x'], expressions: [{ id: 0, op: 'sigmoid', display: 'sigmoid(x)', valueName: 'sigmoid(x)', inputs: ['x'] }] },
    );
    const region = nodes.find(node => node.type === 'regionOp')!;
    expect(opLabelMap[region.id]).toBe('sigmoid(x)');
    expect(ssaMap['%arg0']).toBe('x');
    expect(edges.some(edge => edge.source === '%arg0')).toBe(true);
    expect(nodes.some(node => node.data?.rawOp === 'arith.negf')).toBe(true);
    expect(nodes.some(node => node.data?.rawOp === 'math.exp')).toBe(true);
  });

  test('creates a display-only constants group and keeps constants connected', () => {
    const mlir = `
func.func @forward(%arg0: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0.0 : f32
  %c2 = arith.constant 2.0 : f32
  %r = arith.addf %arg0, %c2 : f32
  return %r : tensor<f32>
}`;
    const { nodes, edges } = parseMLIRToGraph(mlir);
    const group = nodes.find(node => (node.data as any)?.isConstantsGroup);
    expect(group).toBeDefined();
    const constants = nodes.filter(node => node.data?.rawOp === 'arith.constant');
    expect(constants).toHaveLength(2);
    expect(constants.every(node => node.parentId === group!.id)).toBe(true);
    expect(edges.some(edge => edge.source === '%c2')).toBe(true);
  });
});

describe('region input and output zones', () => {
  test('classifies block arguments from ins/outs operands and preserves edges', () => {
    const mlir = `
func.func @forward(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%arg0 : tensor<f32>) {
  ^bb0(%left: f32, %right: f32, %destination: f32):
    %sum = arith.addf %left, %right : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}`;
    const { nodes, edges } = parseMLIRToGraph(mlir);
    const region = nodes.find(node => node.type === 'regionOp')!;
    const children = nodes.filter(node => node.parentId === region.id);
    const inputArgs = children.filter(node => (node.data as any)?.isBlockArg && !(node.data as any).isOutputBlockArg);
    const outputArgs = children.filter(node => (node.data as any)?.isBlockArg && (node.data as any).isOutputBlockArg);
    expect(inputArgs.map(node => (node.data as any).rawLabel)).toEqual(['%left', '%right']);
    expect(outputArgs.map(node => (node.data as any).rawLabel)).toEqual(['%destination']);
    expect(inputArgs.every(node => node.position.y === 58)).toBe(true);
    expect(outputArgs.every(node => node.position.y > 58)).toBe(true);
    expect(edges.some(edge => edge.target === inputArgs[0].id)).toBe(true);
    expect(edges.some(edge => edge.target === inputArgs[1].id)).toBe(true);
    expect(edges.some(edge => edge.target === region.id && edge.targetHandle === 'result')).toBe(true);
  });
});
