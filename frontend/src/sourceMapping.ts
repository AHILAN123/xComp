import type { Node } from '@xyflow/react';

export type SourceExpression = {
  id: number;
  op: string;
  display: string;
  valueName: string;
  inputs: string[];
  assignedName?: string;
};

export type SourceMetadata = {
  args?: string[];
  expressions?: SourceExpression[];
  exprs?: SourceExpression[];
  assignments?: { lhs: string; rhs: string; rhs_idents?: string[] }[];
  ops?: { kind: string; expr: string }[];
};

const COMPOSITE_SOURCE_OPS = new Set([
  'relu', 'sigmoid', 'tanh', 'exp', 'log', 'sqrt', 'abs', 'softmax',
  'gelu', 'silu', 'clamp', 'where', 'maximum', 'minimum',
]);

type ParsedValue = { display: string; key: string; op?: string; inputs?: string[] };

const BINARY: Record<string, string> = {
  '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '@': 'matmul', '%': 'mod', '**': 'pow',
};
const CALL_SYMBOL: Record<string, string> = {
  add: '+', sub: '-', mul: '*', div: '/', matmul: '@', mm: '@', bmm: '@',
};

function splitTopLevel(text: string, separator: string): string[] | null {
  let depth = 0;
  for (let i = 0; i <= text.length - separator.length; i++) {
    const ch = text[i];
    if (ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === ')' || ch === ']' || ch === '}') depth--;
    if (depth === 0 && text.slice(i, i + separator.length) === separator) {
      return [text.slice(0, i).trim(), text.slice(i + separator.length).trim()];
    }
  }
  return null;
}

function splitArguments(text: string): string[] {
  const result: string[] = [];
  let start = 0;
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    if ('([{'.includes(text[i])) depth++;
    else if (')]}'.includes(text[i])) depth--;
    else if (text[i] === ',' && depth === 0) {
      result.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  if (text.slice(start).trim()) result.push(text.slice(start).trim());
  return result;
}

function stripOuterParens(text: string): string {
  let value = text.trim();
  while (value.startsWith('(') && value.endsWith(')')) {
    let depth = 0;
    let closesEarly = false;
    for (let i = 0; i < value.length; i++) {
      if (value[i] === '(') depth++;
      else if (value[i] === ')') depth--;
      if (depth === 0 && i < value.length - 1) { closesEarly = true; break; }
    }
    if (closesEarly) break;
    value = value.slice(1, -1).trim();
  }
  return value;
}

function parseLiteral(text: string): string | undefined {
  const value = text.trim();
  if (/^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    const number = Number(value);
    return Number.isFinite(number) && Number.isInteger(number) ? String(number) : value;
  }
  return undefined;
}

function operationName(callee: string): string {
  return callee.split('.').pop() || callee;
}

function parseSourceExpressions(code: string): { args: string[]; expressions: SourceExpression[] } {
  const args = (code.match(/def\s+forward\s*\(([^)]*)\)/)?.[1] || '')
    .split(',').map(v => v.trim().split(':')[0].split('=')[0].replace(/^\*+/, '').trim())
    .filter(v => v && v !== 'self' && v !== 'cls');
  const expressions: SourceExpression[] = [];
  const known = new Set(args);
  const bindings = new Map<string, string>();
  args.forEach(arg => bindings.set(arg, arg));
  const seenInline = new Map<string, SourceExpression>();
  let nextId = 0;

  const addExpression = (value: ParsedValue, assignedName?: string): ParsedValue => {
    const valueName = assignedName || value.key;
    const existing = !assignedName ? seenInline.get(value.display) : undefined;
    if (existing) return { ...value, key: existing.valueName };
    const expression: SourceExpression = {
      id: nextId++, op: value.op || 'value', display: value.display,
      valueName, inputs: value.inputs || [], ...(assignedName ? { assignedName } : {}),
    };
    expressions.push(expression);
    if (!assignedName) seenInline.set(value.display, expression);
    return { ...value, key: valueName };
  };

  const parse = (raw: string): ParsedValue => {
    let text = stripOuterParens(raw.trim());
    const literal = parseLiteral(text);
    if (literal !== undefined) return { display: literal, key: literal, op: 'constant', inputs: [] };
    if (/^[A-Za-z_]\w*$/.test(text)) {
      // Keep the variable spelling in composed expressions (z * 2), while
      // retaining the binding as the stable source-level value key.
      return { display: text, key: text, inputs: [text] };
    }

    for (const operator of ['**', '@', '+', '-', '*', '/']) {
      const parts = splitTopLevel(text, operator);
      if (parts && parts[0] && parts[1] && !(operator === '-' && parts[0].endsWith('e'))) {
        const left = parse(parts[0]);
        const right = parse(parts[1]);
        const leftDisplay = /[+\-*/@]/.test(left.display) ? `(${left.display})` : left.display;
        const rightDisplay = /[+\-*/@]/.test(right.display) ? `(${right.display})` : right.display;
        const display = `${leftDisplay} ${operator} ${rightDisplay}`;
        const inline = addExpression({ display, key: display, op: BINARY[operator], inputs: [left.key, right.key] });
        return { display, key: inline.key, op: BINARY[operator], inputs: [left.key, right.key] };
      }
    }
    if (text.startsWith('-')) {
      const inner = parse(text.slice(1));
      const display = `-${inner.display}`;
      const inline = addExpression({ display, key: display, op: 'neg', inputs: [inner.key] });
      return { display, key: inline.key, op: 'neg', inputs: [inner.key] };
    }

    const callMatch = text.match(/^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\((.*)\)$/s);
    if (callMatch) {
      const callee = operationName(callMatch[1]);
      const parsedArgs = splitArguments(callMatch[2]).map(parse);
      const symbol = CALL_SYMBOL[callee];
      const display = symbol && parsedArgs.length >= 2
        ? `${parsedArgs[0].display} ${symbol} ${parsedArgs[1].display}`
        : `${callee}(${parsedArgs.map(arg => arg.display).join(', ')})`;
      const inline = addExpression({ display, key: display, op: callee, inputs: parsedArgs.map(arg => arg.key) });
      return { display, key: inline.key, op: callee, inputs: parsedArgs.map(arg => arg.key) };
    }
    return { display: text, key: text, op: 'unknown', inputs: [text] };
  };

  const forwardStart = code.search(/def\s+forward\s*\([^)]*\)\s*:/);
  const body = forwardStart >= 0 ? code.slice(forwardStart).split('\n').slice(1) : code.split('\n');
  for (const rawLine of body) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#') || line.startsWith('class ') || line.startsWith('def ')) continue;
    if (line.startsWith('return ')) {
      parse(line.slice(7));
      continue;
    }
    const assignment = line.match(/^([A-Za-z_]\w*)\s*=\s*(?!=)(.*)$/);
    if (assignment && !['model', 'inputs'].includes(assignment[1])) {
      const parsed = parse(assignment[2]);
      // The root expression is represented by the assigned value, not by a
      // second anonymous SSA value. Keep nested inline expressions intact.
      const rootIndex = expressions.findLastIndex(expression => expression.display === parsed.display && !expression.assignedName);
      if (rootIndex >= 0) {
        seenInline.delete(parsed.display);
        expressions.splice(rootIndex, 1);
      }
      addExpression(parsed, assignment[1]);
      known.add(assignment[1]);
      bindings.set(assignment[1], parsed.display);
      const last = expressions[expressions.length - 1];
      if (last && last.assignedName === assignment[1]) last.inputs = parsed.inputs || [];
    }
  }
  return { args, expressions };
}

function opAffinity(mlirOp: string, childOps: string[], sourceOp: string): number {
  const text = [mlirOp, ...childOps].join(' ');
  const groups: Record<string, RegExp> = {
    matmul: /linalg\.(matmul|batch_matmul)|mm|bmm/,
    add: /addf|addi|linalg\.add/,
    sub: /subf|subi|linalg\.sub/,
    mul: /mulf|muli|linalg\.mul/,
    div: /divf|divsi|divui|linalg\.div/,
    relu: /maxf|cmpf.*select|select.*cmpf/,
  };
  return groups[sourceOp]?.test(text) ? 2 : 0;
}

function parseMlirOps(mlir: string) {
  const lines = mlir.split('\n');
  const result: { results: string[]; op: string; ins: string[]; childOps: string[]; depth: number }[] = [];
  let braceDepth = 0;
  let current: (typeof result)[number] | undefined;
  let funcBodyDepth = -1;
  for (const raw of lines) {
    const line = raw.trim();
    const before = braceDepth;
    if (line.startsWith('func.func') || line.startsWith('func ')) {
      if (line.includes('{')) funcBodyDepth = before + 1;
    }
    const header = line.match(/^((?:%[\w$.-]+\s*,?\s*)+)\s*=\s*([A-Za-z_][\w.]*)/);
    if (header && (funcBodyDepth < 0 || before === funcBodyDepth)) {
      if (current) result.push(current);
      const results = (header[1].match(/%[A-Za-z0-9_$.-]+/g) || []);
      current = { results, op: header[2], ins: [], childOps: [], depth: before };
    } else if (current) {
      const child = line.match(/^(?:%[\w$.-]+\s*=\s*)?([A-Za-z_][\w.]*)/);
      if (before > current.depth && child && !line.startsWith('^')) current.childOps.push(child[1]);
    }
    const ins = line.match(/\bins\s*\(([^)]*)\)/);
    if (ins && current) current.ins.push(...(ins[1].match(/%[A-Za-z0-9_$.-]+/g) || []));
    braceDepth += (line.match(/\{/g) || []).length - (line.match(/\}/g) || []).length;
    if (current && before === current.depth && line.startsWith('return')) { result.push(current); current = undefined; }
  }
  if (current) result.push(current);
  return result.filter(op => funcBodyDepth < 0 || op.depth === funcBodyDepth);
}

export function buildPyTorchDisplayMaps(
  pythonCode: string,
  mlirCode: string,
  nodes: Node[],
  metadata?: SourceMetadata,
): { ssaMap: Record<string, string>; opLabelMap: Record<string, string>; compositeNodeIds: string[] } {
  const source = metadata?.expressions || metadata?.exprs;
  const derived: { args: string[]; expressions: SourceExpression[] } = source?.length
    ? { args: metadata?.args || [], expressions: source }
    : parseSourceExpressions(pythonCode);
  const expressions = derived.expressions;
  const ssaMap: Record<string, string> = {};
  const opLabelMap: Record<string, string> = {};
  const compositeNodeIds = new Set<string>();
  derived.args.forEach((arg, index) => { ssaMap[`%arg${index}`] = arg; });

  const mlirOps = parseMlirOps(mlirCode);
  const nodeByResult = new Map<string, Node>();
  for (const node of nodes) {
    const data = node.data as any;
    const results: string[] = Array.isArray(data?.mlirResults) ? data.mlirResults : [];
    for (const result of results) nodeByResult.set(result, node);
  }
  const sourceUsed = new Set<number>();
  const knownValues = new Map<string, string>();
  for (const [ssa, name] of Object.entries(ssaMap)) knownValues.set(ssa, name);

  const resolve = (ssa: string): string | undefined => knownValues.get(ssa);
  for (const op of mlirOps) {
    if (op.op === 'arith.constant') {
      const node = op.results.map(result => nodeByResult.get(result)).find(Boolean);
      const value = (node?.data as any)?.constValue;
      if (value != null) op.results.forEach(result => knownValues.set(result, String(value)));
      continue;
    }
    if (!op.ins.length) continue;
    const inputNames = op.ins.map(resolve).filter((value): value is string => Boolean(value));
    const candidates = expressions.map((expression, index) => ({ expression, index }))
      .filter(({ expression }) => {
        // Literal scalars may be folded into a region or materialized as a
        // separate arith.constant. They are not required for matching the
        // tensor dataflow, while named/subexpression inputs are.
        const requiredInputs = expression.inputs.filter(input => !/^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(input));
        return requiredInputs.length > 0 && requiredInputs.every(input => inputNames.includes(input));
      })
      .sort((a, b) => {
        const aUsed = sourceUsed.has(a.index) ? 0 : 1;
        const bUsed = sourceUsed.has(b.index) ? 0 : 1;
        return bUsed - aUsed || opAffinity(op.op, op.childOps, b.expression.op) - opAffinity(op.op, op.childOps, a.expression.op) || a.index - b.index;
      });
    const match = candidates[0];
    if (!match) continue;
    sourceUsed.add(match.index);
    const expression = match.expression;
    for (const result of op.results) knownValues.set(result, expression.valueName);
    const node = op.results.map(result => nodeByResult.get(result)).find(Boolean);
    if (node) {
      const label = expression.assignedName ? `${expression.assignedName} = ${expression.display}` : expression.display;
      opLabelMap[node.id] = label;
      if (COMPOSITE_SOURCE_OPS.has(expression.op)) compositeNodeIds.add(node.id);
    }
  }
  for (const node of nodes) {
    const data = node.data as any;
    if (data?.isBlockArg && typeof data.operandName === 'string') {
      const inherited = resolve(data.operandName);
      if (inherited) ssaMap[data.operandName] = inherited;
    }
  }
  return { ssaMap, opLabelMap, compositeNodeIds: [...compositeNodeIds] };
}

export const buildSSADisplayNameMap = buildPyTorchDisplayMaps;
