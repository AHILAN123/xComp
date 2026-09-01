import { type Node, type Edge } from '@xyflow/react';
export { buildPyTorchDisplayMaps, buildSSADisplayNameMap } from './sourceMapping';
export type { SourceMetadata as PyTorchSourceMeta } from './sourceMapping';

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
  insCount?: number;
  outputBlockNodeIds?: string[];
  inputBlockNodeIds?: string[];
  argsMapped?: boolean;
  blockArgs?: number;
  childY: number;
  startY?: number;
}

interface ConstantsGroup {
  id: string;
  count: number;
  node: Node;
}

interface DisplayGroup {
  id: string;
  count: number;
  node: Node;
}

const SPAWN_JITTER = 14;

function jitterPosition(position: { x: number; y: number }): { x: number; y: number } {
  return {
    x: position.x + (Math.random() * 2 - 1) * SPAWN_JITTER,
    y: position.y + (Math.random() * 2 - 1) * SPAWN_JITTER,
  };
}

function numericDimension(value: unknown, fallback: number): number {
  const dimension = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(dimension) && dimension > 0 ? dimension : fallback;
}

function nodeDimension(node: Node, axis: 'width' | 'height', fallback: number): number {
  const measured = node.measured?.[axis];
  const configured = node[axis];
  const styled = node.style?.[axis];
  return numericDimension(measured ?? configured ?? styled, fallback);
}

function distributeHorizontally(group: Node[], availableWidth: number, minimumGap = 16, padding = 16): void {
  if (group.length === 0) return;
  const widths = group.map((node) => nodeDimension(node, 'width', 100));
  const totalWidth = widths.reduce((sum, width) => sum + width, 0);
  const usableWidth = Math.max(0, availableWidth - padding * 2);
  const gap = Math.max(0, Math.min(minimumGap, (usableWidth - totalWidth) / Math.max(1, group.length - 1)));
  const occupiedWidth = totalWidth + gap * Math.max(0, group.length - 1);
  let x = padding + Math.max(0, (usableWidth - occupiedWidth) / 2);
  group.forEach((node, index) => {
    node.position = { x: Math.min(Math.max(padding, x), Math.max(padding, availableWidth - padding - widths[index])), y: node.position.y };
    x += widths[index] + gap;
  });
}

function placeTensorSetupNodes(nodes: Node[], inputSection: Node | undefined): void {
  const setupNodes = nodes
    .map((node, index) => ({ node, index }))
    .filter(({ node }) => !node.parentId && (node.data?.rawOp === 'tensor.empty' || node.data?.rawOp === 'linalg.fill'))
    .sort((a, b) => {
      const rank = (rawOp: string) => rawOp === 'tensor.empty' ? 0 : 1;
      return rank(String(a.node.data.rawOp)) - rank(String(b.node.data.rawOp)) || a.index - b.index;
    })
    .map(({ node }) => node);
  if (setupNodes.length === 0) return;

  const inputTop = inputSection?.position.y ?? 0;
  const inputHeight = nodeDimension(inputSection as Node ?? {}, 'height', 120);
  const startY = inputTop + inputHeight + 32;
  const availableWidth = 430;
  const gap = 24;
  const widths = setupNodes.map((node) => nodeDimension(node, 'width', 100));
  const totalWidth = widths.reduce((sum, width) => sum + width, 0) + gap * (setupNodes.length - 1);
  let x = Math.max(16, (availableWidth - totalWidth) / 2);
  setupNodes.forEach((node, index) => {
    const width = widths[index];
    node.position = clampPosition(jitterPosition({ x, y: startY + index * 92 }), {
      minX: 16,
      maxX: Math.max(16, availableWidth - 16 - width),
      minY: startY,
      maxY: startY + index * 92,
    });
    x += width + gap;
  });
}

function clampPosition(position: { x: number; y: number }, bounds: { minX: number; maxX: number; minY: number; maxY: number }) {
  return {
    x: Math.min(bounds.maxX, Math.max(bounds.minX, position.x)),
    y: Math.min(bounds.maxY, Math.max(bounds.minY, position.y)),
  };
}

const EDGE_STYLES = {
  operand: { stroke: 'var(--graph-edge)', strokeWidth: 1.6 },
  blockArg: { stroke: 'var(--graph-edge-block)', strokeDasharray: '4 3', strokeWidth: 1.4 },
  yield: { stroke: 'var(--graph-edge-yield)', strokeDasharray: '6 4', strokeWidth: 1.6 },
  ret: { stroke: 'var(--graph-edge-return)', strokeWidth: 2 },
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

const ARITH_OP_DISPLAY: Record<string, string> = {
  'arith.addf': '+', 'arith.addi': '+', 'arith.subf': '-', 'arith.subi': '-',
  'arith.mulf': '*', 'arith.muli': '*', 'arith.divf': '/', 'arith.divsi': '/',
  'arith.divui': '/', 'arith.negf': 'unary -', 'arith.negi': 'unary -',
  'math.powf': '^', 'math.powi': '^', 'powf': '^', 'powi': '^',
};

function parseConstValue(line: string): string | null {
  const match = line.match(/arith\.constant\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) ? String(value) : null;
}

// ssaMap: NodeId → PyTorch display name
// opLabelMap: NodeId (of container/op node) → display label in PyTorch mode
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

  const base = { background: 'var(--graph-node-bg)', border: '1px solid var(--graph-node-border)', borderRadius: '10px', fontFamily: 'ui-monospace, monospace', fontSize: '11px', padding: '8px 10px', color: 'var(--graph-node-text)' };

  const nodeIds = new Set<string>();
  const uniqueId = (base0: string) => { let id = base0; let n = 1; while (nodeIds.has(id)) id = `${base0}#${++n}`; nodeIds.add(id); return id; };
  let constantsGroup: ConstantsGroup | undefined;
  let outputsGroup: DisplayGroup | undefined;
  let inputsSection: Node | undefined;

  const addConstantToGroup = (node: Node, value: string | null) => {
    if (!value || node.parentId) return;
    if (!constantsGroup) {
      const id = uniqueId('section_constants');
      const group: Node = {
        id,
        type: 'group',
        data: { label: 'CONSTANTS', isConstantsGroup: true },
        position: jitterPosition({ x: COL.ops, y: -150 }),
        draggable: true,
        selectable: false,
        style: {
          width: 190,
          height: 100,
          background: 'var(--graph-constants-bg)',
          border: '1px dashed var(--graph-constants-border)',
          borderRadius: '12px',
          fontFamily: 'ui-monospace, monospace',
          fontSize: '11px',
          color: 'var(--graph-constants-text)',
        },
      };
      nodes.push(group);
      constantsGroup = { id, count: 0, node: group };
    }
    const index = constantsGroup.count++;
    node.parentId = constantsGroup.id;
    node.extent = 'parent';
    node.position = jitterPosition({ x: 15 + index * 76, y: 42 });
    node.draggable = true;
    node.style = {
      ...node.style,
      width: 'auto',
      minWidth: '42px',
      textAlign: 'center',
      whiteSpace: 'nowrap',
       background: 'var(--graph-constants-bg)',
       borderColor: 'var(--graph-constants-border)',
    };
    constantsGroup.node.style = {
      ...constantsGroup.node.style,
      width: Math.max(190, 30 + constantsGroup.count * 76),
    };
  };

  const pushEdge = (source: string, target: string, style: any, opts: { animated?: boolean; sourceHandle?: string; targetHandle?: string; type?: string } = {}) => {
    edges.push({ id: `e${edges.length}`, source, target, style, animated: opts.animated, type: opts.type, ...(opts.sourceHandle ? { sourceHandle: opts.sourceHandle } : {}), ...(opts.targetHandle ? { targetHandle: opts.targetHandle } : {}) });
  };

  const operandEdges = (operands: string[], targetId: string, targetHandle?: string, orderedHandles?: string[]) => {
    operands.forEach((name, index) => {
      const src = resolve(name);
      const handle = orderedHandles?.[index] ?? targetHandle;
      if (src && src !== targetId) pushEdge(src, targetId, EDGE_STYLES.operand, { animated: true, ...(handle ? { targetHandle: handle } : {}) });
    });
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
      // For structured operations, block arguments correspond to the actual
      // operand groups, not to their spelling. A trailing output mapping is
      // used when an operation exposes fewer block arguments than operands.
      const operandOffset = Math.max(0, (ctx.operands?.length ?? 0) - names.length);
      names.forEach((name, i) => {
        const id = uniqueId(`${ctx.opNodeId}::${name}`);
        define(name, id);
        const operandIndex = operandOffset + i;
        const operandName = ctx.operands?.[operandIndex];
        const isOutputBlockArg = operandIndex >= (ctx.insCount ?? Number.MAX_SAFE_INTEGER);
        // Region inputs form a vertical hand-off from the green input zone into the body.
        // Keep output block arguments on the existing side-oriented route.
        nodes.push({
          id,
          type: 'default',
           position: jitterPosition({ x: 20 + i * 150, y: ctx.childY }),
          data: {
            label: name,
            rawLabel: name,
            isBlockArg: true,
            zone: isOutputBlockArg ? 'output' : 'input',
            blockArgIndex: i,
            operandName,
            isOutputBlockArg,
          },
          parentId: ctx.opNodeId,
          extent: 'parent',
          draggable: true,
          targetPosition: isOutputBlockArg ? ('left' as any) : ('top' as any),
          sourcePosition: isOutputBlockArg ? ('right' as any) : ('bottom' as any),
           style: { ...base, background: 'var(--graph-block-bg)', borderColor: 'var(--graph-block-border)' }
        });
        const blockNode = nodes[nodes.length - 1];
        if (isOutputBlockArg) {
          (ctx.outputBlockNodeIds ??= []).push(blockNode.id);
        } else {
          (ctx.inputBlockNodeIds ??= []).push(blockNode.id);
        }
        // Enclosing operation operand → region block argument (Relationship 3)
        if (!ctx.argsMapped && ctx.operands) {
          const src = operandName ? resolve(operandName) : undefined;
          // Target the block arg. This is the ONE semantic edge. Force smoothstep logic for 90-degree cleanly separated entry
          if (src && src !== id) pushEdge(src, id, EDGE_STYLES.blockArg, { type: 'smoothstep' });
        }
      });
      // Draw any remaining operands to the container itself (unmapped)
      if (!ctx.argsMapped && ctx.operands && ctx.operands.length > names.length) {
        const unmappedOperands = ctx.operands.slice(0, Math.max(0, (ctx.insCount ?? ctx.operands.length) - names.length));
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

        const h = Math.max(220, ctx.childY + 24);
        const w = Math.max(430, 24 + (ctx.blockArgs ?? 0) * 150);
        const inputY = 58;
        const outputY = Math.max(inputY + 72, h - 62);
        const inputNodes = new Set(ctx.inputBlockNodeIds ?? []);
        const outputNodes = new Set(ctx.outputBlockNodeIds ?? []);
        const directChildren = nodes.filter((child) => child.parentId === ctx.opNodeId);
        distributeHorizontally(directChildren.filter((child) => inputNodes.has(child.id)), w, 16, 18);
        distributeHorizontally(directChildren.filter((child) => outputNodes.has(child.id)), w, 16, 18);
        const bodyGroups = new Map<number, Node[]>();
        directChildren
          .filter((child) => !inputNodes.has(child.id) && !outputNodes.has(child.id))
          .forEach((child) => {
            const layer = child.position.y;
            const group = bodyGroups.get(layer) ?? [];
            group.push(child);
            bodyGroups.set(layer, group);
          });
        bodyGroups.forEach((group) => distributeHorizontally(group, w, 16, 18));
        for (const child of nodes) {
          if (child.parentId !== ctx.opNodeId) continue;
           const childWidth = nodeDimension(child, 'width', 100);
           const childHeight = nodeDimension(child, 'height', 40);
           const zone = inputNodes.has(child.id) ? 'input' : outputNodes.has(child.id) ? 'output' : 'body';
           const zoneTop = zone === 'input' ? inputY : zone === 'output' ? outputY : 37 + (inputNodes.size > 0 ? 70 : 0);
           const zoneBottom = zone === 'input' ? inputY + 70 : zone === 'output' ? h : h - (outputNodes.size > 0 ? 58 : 0);
           const positioned = inputNodes.has(child.id)
             ? { ...child.position, y: inputY }
             : outputNodes.has(child.id)
               ? { ...child.position, y: outputY }
               : child.position;
           child.position = clampPosition(positioned, {
             minX: 0,
             maxX: Math.max(0, w - childWidth),
             minY: zoneTop,
             maxY: Math.max(zoneTop, zoneBottom - childHeight),
           });
        }
        ctx.containerNode.data = {
          ...ctx.containerNode.data,
          inputZoneCount: inputNodes.size,
          outputZoneCount: outputNodes.size,
          headerHeight: 37,
          inputZoneHeight: inputNodes.size > 0 ? 70 : 0,
          outputZoneHeight: outputNodes.size > 0 ? 58 : 0,
        };
        ctx.containerNode.style = { ...ctx.containerNode.style, width: w, height: h };
        const parent = stack[stack.length - 1];
        if (parent.kind === 'region') parent.childY = (ctx.startY ?? 0) + h + 40;
        else yRegions += h + 110;
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
         data: { label: 'INPUTS', isInputsGroup: true },
         position: jitterPosition({ x: COL.args - 20, y: Math.max(0, yArgs - 20) }),
        draggable: false,
        selectable: false,
        style: {
          width: 140,
          height: sectionHeight,
            background: 'var(--graph-input-zone-bg)',
            border: '1px dashed var(--graph-input-zone-border)',
          borderRadius: '12px',
          fontFamily: 'ui-monospace, monospace',
          fontSize: '11px',
            color: 'var(--graph-input-zone-text)',
        },
      });
      inputsSection = nodes[nodes.length - 1];

      funcArgs.forEach((name, i) => {
        const id = uniqueId(name);
        define(name, id);
        nodes.push({
          id,
          type: 'input',
          position: jitterPosition({ x: 15, y: 35 + i * 60 }),
          parentId: inputsSectionId,
          extent: 'parent',
          draggable: false,
          data: { label: name, isFuncArg: true, argIndex: i },
          style: {
            ...base,
             background: 'var(--graph-input-bg)',
             borderColor: 'var(--graph-input-border)',
            width: 'auto',
            minWidth: '36px',
            textAlign: 'center',
            whiteSpace: 'nowrap',
          },
        });
        yArgs += 60;
      });
      distributeHorizontally(nodes.filter((node) => node.parentId === inputsSectionId), 140, 12, 12);
      yArgs += 40;
      stack.push({ kind: 'func', childY: 0 });
      continue;
    }

    if (/^(?:builtin\.)?module\b/.test(line)) { stack.push({ kind: 'module', childY: 0 }); continue; }

    const retMatch = line.match(/^(?:func\.)?return\b/);
    if (retMatch) {
      const names = ssaNames(stripAttributeDicts(line));
      const id = uniqueId(`ret_${edges.length}`);
      if (!outputsGroup) {
        const groupId = uniqueId('section_outputs');
        const group: Node = {
          id: groupId,
          type: 'group',
          data: { label: 'OUTPUTS', isOutputsGroup: true },
           position: jitterPosition({ x: COL.ret - 20, y: yOps }),
          draggable: true,
          selectable: false,
          style: {
            width: 150,
            height: 100,
             background: 'var(--graph-output-zone-bg)',
             border: '1px dashed var(--graph-output-zone-border)',
            borderRadius: '12px',
            fontFamily: 'ui-monospace, monospace',
            fontSize: '11px',
             color: 'var(--graph-output-zone-text)',
          },
        };
        nodes.push(group);
        outputsGroup = { id: groupId, count: 0, node: group };
      }
      const outputIndex = outputsGroup.count++;
      const outputNode: Node = {
        id,
        type: 'output',
        position: jitterPosition({ x: 15 + outputIndex * 90, y: 42 }),
        parentId: outputsGroup.id,
        extent: 'parent',
        data: { label: 'return', rawLabel: 'return' },
         style: { ...base, background: 'var(--graph-output-bg)', borderColor: 'var(--graph-output-border)' },
      };
      nodes.push(outputNode);
      outputsGroup.node.style = { ...outputsGroup.node.style, width: Math.max(150, 30 + outputsGroup.count * 90) };
      distributeHorizontally(
        nodes.filter((node) => node.parentId === outputsGroup!.id),
        Number(outputsGroup.node.style?.width ?? 150), 12, 12,
      );
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
      const yieldLabel = line.split(/\s+/)[0];
      nodes.push({ id, type: 'default', position: jitterPosition({ x: 40, y: ctx.childY }), data: { label: yieldLabel, rawLabel: yieldLabel, isYield: true }, parentId: ctx.opNodeId, extent: 'parent', style: { ...base, background: 'var(--graph-constants-bg)', borderColor: 'var(--graph-constants-border)' } });
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

      const parentCtx = stack[stack.length - 1];
      const isNested = parentCtx.kind === 'region';

      const id = uniqueId(`op_${nodes.length}`);
      for (const name of resultNames) define(name, id); // Results live in enclosing scope

      const node: Node = {
        id,
        type: 'regionOp',
        data: {
          label: resultNames.length ? `${resultNames.join(', ')} = ${opName}` : opName,
          rawLabel: resultNames.length ? `${resultNames.join(', ')} = ${opName}` : opName,
          rawOp: opName,
          mlirResults: resultNames,
        },
        position: jitterPosition(isNested ? { x: 40, y: parentCtx.childY } : { x: COL.regions, y: yRegions }),
        style: { width: 430, height: 300 },
      };
      if (isNested) { node.parentId = parentCtx.opNodeId; node.extent = 'parent'; }

      nodes.push(node);
      const groups = extractOperandGroups(rhs);
      const ctx: Ctx = { kind: 'region', opNodeId: id, containerNode: node, operands: groups.operands, insCount: groups.ins.length, childY: 58, startY: isNested ? parentCtx.childY : undefined };
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

      // Extra metadata for PyTorch name mode
      const arithOp = ARITH_OP_DISPLAY[opName] ?? null;
      const orderedBinary = Boolean(arithOp && operands.length === 2);
      const constVal = opName === 'arith.constant' ? parseConstValue(line) : null;

      const node: Node = {
        id, type: 'default',
        position: jitterPosition(isRegionLocal ? { x: 40, y: parentCtx.childY } : { x: COL.ops, y: yOps }),
        data: {
          label: `${displayName} = ${opName}`,
          rawLabel: `${displayName} = ${opName}`,
          rawOp: opName,
          arithOp,
          orderedBinary,
          constValue: constVal,
          mlirResults: resultNames,
        },
        style: { ...base },
      };
      if (isRegionLocal) { node.parentId = parentCtx.opNodeId; node.extent = 'parent'; parentCtx.childY += 92; }
      else yOps += 145;
      // Assign the visual parent before inserting the node. The constants
      // group is presentation-only; operand edges still point at this node.
      if (opName === 'arith.constant') addConstantToGroup(node, constVal);
      nodes.push(node);

      operandEdges(operands, id, undefined, orderedBinary ? ['operand-0', 'operand-1'] : undefined); // Region-local SSA value → operation / Function arg → operation operand
      continue;
    }
  }

  placeTensorSetupNodes(nodes, inputsSection);
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

function extractOperandGroups(rhs: string): { operands: string[]; ins: string[]; outs: string[] } {
  const cleaned = stripAttributeDicts(rhs);
  const insMatch = cleaned.match(/\bins\s*\(([^)]*)\)/);
  const outsMatch = cleaned.match(/\bouts\s*\(([^)]*)\)/);
  const ins = insMatch ? ssaNames(insMatch[1]) : [];
  const outs = outsMatch ? ssaNames(outsMatch[1]) : [];
  return { operands: [...ins, ...outs], ins, outs };
}
