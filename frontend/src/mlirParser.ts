import { type Node, type Edge } from '@xyflow/react';

const SSA_NAME = /%[A-Za-z0-9_$.-]+/g;

function ssaNames(text: string): string[] {
  return text.match(SSA_NAME) ?? [];
}

function stripAttributeDicts(text: string): string {
  let out = '';
  let depth = 0;
  for (const ch of text) {
    if (ch === '{') { depth++; continue; }
    if (ch === '}') { depth = Math.max(0, depth - 1); continue; }
    if (depth === 0) out += ch;
  }
  return out;
}

interface Ctx {
  kind: 'module' | 'func' | 'region';
  opNodeId?: string;
  containerNode?: Node;
  operands?: string[];
  argsMapped?: boolean;
  blockArgs?: number;
  childY: number;
  startY?: number;
}

const EDGE_STYLES = {
  operand: { stroke: '#71717a', strokeWidth: 1.6 },
  blockArg: { stroke: '#3b82f6', strokeDasharray: '4 3', strokeWidth: 1.4 },
  yield: { stroke: '#f59e0b', strokeDasharray: '6 4', strokeWidth: 1.6 },
  ret: { stroke: '#18181b', strokeWidth: 2 },
} as const;

export function extractPyTorchArgsFromPython(code: string): string[] {
  if (!code) return [];
  const match = code.match(/def\s+forward\s*\(\s*([^)]*)\)/);
  if (!match) return [];
  const rawArgs = match[1].split(',');
  const result: string[] = [];
  for (const raw of rawArgs) {
    const trimmed = raw.trim();
    if (!trimmed) continue;
    let name = trimmed.split(':')[0].split('=')[0].trim();
    name = name.replace(/^[*]+/, '');
    if (name && name !== 'self' && name !== 'cls') {
      result.push(name);
    }
  }
  return result;
}

export function buildSSADisplayNameMap(pythonCode: string, mlirCode: string, nodes: Node[]): Record<string, string> {
  const ssaMap: Record<string, string> = {};

  // 1. Root: Function Arguments (%arg0, %arg1, ...)
  const pyArgs = extractPyTorchArgsFromPython(pythonCode);
  const funcArgNodes = nodes.filter(n => n.type === 'input' || n.data?.isFuncArg);
  funcArgNodes.sort((a, b) => (a.data?.argIndex ?? 0) - (b.data?.argIndex ?? 0));

  funcArgNodes.forEach((node, i) => {
    if (pyArgs[i]) {
      ssaMap[node.id] = pyArgs[i];
    }
  });

  // 2. Extract Python Assignment Statements (e.g. y = torch.matmul(x, w))
  const pyAssignments: { lhs: string; rhsIdents: string[] }[] = [];
  if (pythonCode) {
    for (const rawLine of pythonCode.split('\n')) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#') || line.startsWith('def ') || line.startsWith('class ')) continue;
      const eqIdx = line.indexOf('=');
      if (eqIdx > 0 && !line.includes('==') && !line.includes('!=') && !line.includes('<=')) {
        const lhs = line.slice(0, eqIdx).trim();
        const rhs = line.slice(eqIdx + 1).trim();
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(lhs) && lhs !== 'model' && lhs !== 'inputs') {
          const idents = Array.from(new Set(rhs.match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) || []))
            .filter(id => id !== 'torch' && id !== 'nn' && id !== 'F' && id !== 'self' && id !== lhs);
          pyAssignments.push({ lhs, rhsIdents: idents });
        }
      }
    }
  }

  // 3. Match SSA Operation Output Values with Python Assignments
  if (mlirCode) {
    for (const rawLine of mlirCode.split('\n')) {
      const line = rawLine.trim();
      if (line.startsWith('%') && line.includes('=')) {
        const eqIdx = line.indexOf('=');
        const lhsResults = line.slice(0, eqIdx).split(',').map(s => s.trim()).filter(Boolean);
        const rhs = line.slice(eqIdx + 1).trim();

        const operands = (rhs.match(/%[A-Za-z0-9_$.-]+/g) || []).filter(op => !lhsResults.includes(op));
        const operandDisplayNames = operands.map(op => ssaMap[op]).filter(Boolean);

        for (const assign of pyAssignments) {
          const matchCount = assign.rhsIdents.filter(ident => operandDisplayNames.includes(ident)).length;
          if (matchCount > 0) {
            for (const res of lhsResults) {
              if (!ssaMap[res]) {
                ssaMap[res] = assign.lhs;
              }
            }
            break;
          }
        }
      }
    }
  }

  return ssaMap;
}

export function parseMLIRToGraph(code: string): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  const scopes: Map<string, string>[] = [new Map()];
  const stack: Ctx[] = [{ kind: 'module', childY: 0 }];

  const define = (name: string, nodeId: string) => { scopes[scopes.length - 1].set(name, nodeId); };
  const resolve = (name: string): string | undefined => {
    for (let i = scopes.length - 1; i >= 0; i--) { const id = scopes[i].get(name); if (id) return id; }
    return undefined;
  };

  const COL = { args: 40, ops: 320, regions: 640, ret: 1080 };
  let yArgs = 0, yOps = 0, yRegions = 0;

  const base = { background: '#ffffff', border: '1px solid #e4e4e7', borderRadius: '10px', fontFamily: 'ui-monospace, monospace', fontSize: '11px', padding: '8px 10px' };

  const nodeIds = new Set<string>();
  const uniqueId = (base0: string) => { let id = base0; let n = 1; while (nodeIds.has(id)) id = `${base0}#${++n}`; nodeIds.add(id); return id; };

  const pushEdge = (source: string, target: string, style: any, opts: { animated?: boolean; sourceHandle?: string; targetHandle?: string; type?: string } = {}) => {
    edges.push({ id: `e${edges.length}`, source, target, style, animated: opts.animated, type: opts.type, ...(opts.sourceHandle ? { sourceHandle: opts.sourceHandle } : {}), ...(opts.targetHandle ? { targetHandle: opts.targetHandle } : {}) });
  };

  const operandEdges = (operands: string[], targetId: string, targetHandle?: string) => {
    for (const name of operands) {
      const src = resolve(name);
      if (src && src !== targetId) pushEdge(src, targetId, EDGE_STYLES.operand, { animated: true, ...(targetHandle ? { targetHandle } : {}) });
    }
  };

  const lines = code.split('\n');

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;

    // Block header: ^bb0(%in: f32, %in_0: f32, %out: f32):
    const blockMatch = line.match(/^\^[A-Za-z0-9_]+\s*\(([^)]*)\)\s*:/);
    if (blockMatch) {
      const ctx = stack[stack.length - 1];
      if (ctx.kind !== 'region') continue;
      const names = ssaNames(blockMatch[1]);
      ctx.blockArgs = names.length;
      names.forEach((name, i) => {
        const id = uniqueId(`${ctx.opNodeId}::${name}`);
        define(name, id);
        const operandName = ctx.operands ? ctx.operands[i] : undefined;
        const isOutputBlockArg = i === names.length - 1 || name.includes('out');
        // Force the targetPosition to Left for block arguments to guarantee clean orthogonal routing
        nodes.push({
          id,
          type: 'default',
          position: { x: 20 + i * 150, y: ctx.childY },
          data: {
            label: name,
            rawLabel: name,
            isBlockArg: true,
            blockArgIndex: i,
            operandName,
            isOutputBlockArg,
          },
          parentId: ctx.opNodeId,
          extent: 'parent',
          targetPosition: 'left' as any,
          sourcePosition: 'bottom' as any,
          style: { ...base, background: '#eef2ff', borderColor: '#c7d2fe' }
        });
        // Enclosing operation operand → region block argument (Relationship 3)
        if (!ctx.argsMapped && ctx.operands) {
          const src = operandName ? resolve(operandName) : undefined;
          // Target the block arg. This is the ONE semantic edge. Force smoothstep logic for 90-degree cleanly separated entry
          if (src && src !== id) pushEdge(src, id, EDGE_STYLES.blockArg, { type: 'smoothstep' });
        }
      });
      // Draw any remaining operands to the container itself (unmapped)
      if (!ctx.argsMapped && ctx.operands && ctx.operands.length > names.length) {
        const unmappedOperands = ctx.operands.slice(names.length);
        operandEdges(unmappedOperands, ctx.opNodeId!, 'in');
      }
      ctx.argsMapped = true;
      if (names.length > 0) ctx.childY += 90;
      continue;
    }

    if (line.startsWith('}')) {
      const ctx = stack.pop()!;
      if (ctx.kind === 'region' && ctx.containerNode) {
        // If we closed the region without ever mapping operands to a block header,
        // draw the operand edges to the container itself.
        if (!ctx.argsMapped && ctx.operands) {
          operandEdges(ctx.operands, ctx.opNodeId!, 'in');
        }

        const h = Math.max(160, ctx.childY + 24);
        const w = Math.max(430, 24 + (ctx.blockArgs ?? 0) * 150);
        ctx.containerNode.style = { ...ctx.containerNode.style, width: w, height: h };
        const parent = stack[stack.length - 1];
        if (parent.kind === 'region') parent.childY = (ctx.startY ?? 0) + h + 40;
        else yRegions += h + 60;
      }
      if (ctx.kind === 'func' || ctx.kind === 'region') scopes.pop();
      continue;
    }

    const funcMatch = line.match(/^(?:func\.func|func)\s+@[A-Za-z0-9_]+\s*\(([^)]*)\)/);
    if (funcMatch) {
      scopes.push(new Map());
      const funcArgs = ssaNames(funcMatch[1]);
      const inputsSectionId = uniqueId('section_inputs');
      const sectionHeight = Math.max(120, 45 + funcArgs.length * 60);

      nodes.push({
        id: inputsSectionId,
        type: 'group',
        data: { label: 'Inputs' },
        position: { x: COL.args - 20, y: Math.max(0, yArgs - 20) },
        draggable: false,
        selectable: false,
        style: {
          width: 140,
          height: sectionHeight,
          background: 'rgba(241, 245, 249, 0.4)',
          border: '1px dashed #cbd5e1',
          borderRadius: '12px',
          fontFamily: 'ui-monospace, monospace',
          fontSize: '11px',
          color: '#64748b',
        },
      });

      funcArgs.forEach((name, i) => {
        const id = uniqueId(name);
        define(name, id);
        nodes.push({
          id,
          type: 'input',
          position: { x: 15, y: 35 + i * 60 },
          parentId: inputsSectionId,
          extent: 'parent',
          draggable: false,
          data: { label: name, isFuncArg: true, argIndex: i },
          style: {
            ...base,
            background: '#eff6ff',
            borderColor: '#bfdbfe',
            width: 'auto',
            minWidth: '36px',
            textAlign: 'center',
            whiteSpace: 'nowrap',
          },
        });
        yArgs += 60;
      });
      yArgs += 40;
      stack.push({ kind: 'func', childY: 0 });
      continue;
    }

    if (/^(?:builtin\.)?module\b/.test(line)) { stack.push({ kind: 'module', childY: 0 }); continue; }

    const retMatch = line.match(/^(?:func\.)?return\b/);
    if (retMatch) {
      const names = ssaNames(stripAttributeDicts(line));
      const id = uniqueId(`ret_${edges.length}`);
      nodes.push({ id, type: 'output', position: { x: COL.ret, y: yOps + 40 }, data: { label: 'return' }, style: { ...base, background: '#f4f4f5', borderColor: '#a1a1aa' } });
      for (const name of names) {
        const src = resolve(name);
        if (src) pushEdge(src, id, EDGE_STYLES.ret, { animated: true }); // Operation result → return (Relationship 6)
      }
      continue;
    }

    if (/^(?:[a-z_][a-z0-9_]*\.)?yield\b/.test(line)) {
      const ctx = stack[stack.length - 1];
      if (ctx.kind !== 'region') continue;
      const names = ssaNames(stripAttributeDicts(line));
      const id = uniqueId(`${ctx.opNodeId}::yield`);
      nodes.push({ id, type: 'default', position: { x: 40, y: ctx.childY }, data: { label: line.split(/\s+/)[0] }, parentId: ctx.opNodeId, extent: 'parent', style: { ...base, background: '#fffbeb', borderColor: '#fcd34d' } });
      for (const name of names) {
        const src = resolve(name);
        if (src && src !== id) pushEdge(src, id, EDGE_STYLES.operand); // Region-local SSA value → terminator
      }
      // Remap container's result names to this yield node so external consumers connect to yield
      if (ctx.containerNode) {
        const label = ctx.containerNode.data.label as string;
        const results = label.split('=')[0].split(',').map(s => s.trim()).filter(Boolean);
        for (const r of results) define(r, id);
      }
      pushEdge(id, ctx.opNodeId!, EDGE_STYLES.yield, { targetHandle: 'result', type: 'smoothstep' }); // Internal yield → container right-side output vertex
      ctx.childY += 90;
      continue;
    }

    const openBraces = (line.match(/\{/g) ?? []).length;
    const closeBraces = (line.match(/\}/g) ?? []).length;
    const netBrace = openBraces - closeBraces;

    if (line.startsWith('%') && netBrace === 1) {
      // Region-holding operation
      const eq = line.indexOf('=');
      const resultNames = line.slice(0, eq).split(',').map(s => s.trim()).filter(Boolean);
      const rhs = line.slice(eq + 1);
      const opName = rhs.trim().match(/^([A-Za-z_][A-Za-z0-9_.]*)/)?.[1] ?? 'op';
      const operands = extractOperands(rhs, resultNames);

      const parentCtx = stack[stack.length - 1];
      const isNested = parentCtx.kind === 'region';

      const id = uniqueId(`op_${nodes.length}`);
      for (const name of resultNames) define(name, id); // Results live in enclosing scope

      const node: Node = {
        id,
        type: 'regionOp',
        data: { label: resultNames.length ? `${resultNames.join(', ')} = ${opName}` : opName },
        position: isNested ? { x: 40, y: parentCtx.childY } : { x: COL.regions, y: yRegions },
        style: { width: 430, height: 300 },
      };
      if (isNested) { node.parentId = parentCtx.opNodeId; node.extent = 'parent'; }

      nodes.push(node);
      const ctx: Ctx = { kind: 'region', opNodeId: id, containerNode: node, operands, childY: 58, startY: isNested ? parentCtx.childY : undefined };
      if (isNested) parentCtx.childY += 320;
      stack.push(ctx);
      scopes.push(new Map());

      // We defer drawing operand edges until the block header to avoid drawing
      // duplicate edges to both the container AND the block argument.
      continue;
    }

    if (line.startsWith('%') && line.includes('=')) {
      // Simple operation
      const eq = line.indexOf('=');
      const resultNames = line.slice(0, eq).split(',').map(s => s.trim()).filter(Boolean);
      const rhs = line.slice(eq + 1);
      const opName = rhs.trim().match(/^([A-Za-z_][A-Za-z0-9_.]*)/)?.[1] ?? 'op';
      const operands = extractOperands(rhs, resultNames);

      const parentCtx = stack[stack.length - 1];
      const isRegionLocal = parentCtx.kind === 'region';

      const displayName = resultNames[0] ?? opName;
      const id = isRegionLocal ? uniqueId(`${parentCtx.opNodeId}::${displayName}`) : uniqueId(displayName);
      for (const name of resultNames) define(name, id);

      const node: Node = { id, type: 'default', position: isRegionLocal ? { x: 40, y: parentCtx.childY } : { x: COL.ops, y: yOps }, data: { label: `${displayName} = ${opName}` }, style: { ...base } };
      if (isRegionLocal) { node.parentId = parentCtx.opNodeId; node.extent = 'parent'; parentCtx.childY += 92; }
      else yOps += 120;
      nodes.push(node);

      operandEdges(operands, id); // Region-local SSA value → operation / Function arg → operation operand
      continue;
    }
  }

  return { nodes, edges };
}

function extractOperands(rhs: string, results: string[]): string[] {
  const cleaned = stripAttributeDicts(rhs);
  const ins = cleaned.match(/\bins\s*\(([^)]*)\)/);
  const outs = cleaned.match(/\bouts\s*\(([^)]*)\)/);
  if (ins || outs) {
    const list: string[] = [];
    if (ins) list.push(...ssaNames(ins[1]));
    if (outs) list.push(...ssaNames(outs[1]));
    return list;
  }
  return ssaNames(cleaned).filter((n) => !results.includes(n));
}
