import { useEffect, useState, useMemo, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { ReactFlow, Background, Controls, Handle, Position, useNodesState, useEdgesState, type Node, type Edge, type NodeChange } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { FileCode, Activity, Maximize2, Play, Code2, Sun, Moon, Minimize2 } from 'lucide-react';
import { parseMLIRToGraph, extractPyTorchArgsFromPython } from './mlirParser';
import { buildPyTorchDisplayMaps, type SourceMetadata } from './sourceMapping';

function RegionOpNode({ data }: { data: any }) {
  return (
    <div title={`PyTorch: ${data.label}\nMLIR: ${data.rawLabel || data.label}`} className="relative h-full w-full rounded-xl border border-[var(--graph-container-border)] bg-[var(--graph-container-bg)] shadow-sm overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-[37px] bottom-0 flex flex-col">
        {data.inputZoneCount > 0 && <div className="h-[70px] shrink-0 border-b border-[var(--graph-input-zone-border)] bg-[var(--graph-input-zone-bg)]" />}
        <div className="min-h-0 flex-1 bg-[var(--graph-container-inner)]" />
        {data.outputZoneCount > 0 && <div className="h-[58px] shrink-0 border-t border-[var(--graph-output-zone-border)] bg-[var(--graph-output-zone-bg)]" />}
      </div>
      <div className="px-3 py-2 bg-[var(--graph-header-bg)] border-b border-[var(--graph-container-border)]">
        <div className="flex items-center gap-2">
           <span className="text-[11px] font-mono text-[var(--graph-node-text)] truncate block flex-1">{data.label}</span>
          {data.isComposite && (
            <button
              type="button"
              aria-label={data.collapsed ? 'Expand implementation' : 'Collapse implementation'}
              title={data.collapsed ? 'Expand implementation' : 'Collapse implementation'}
              className="shrink-0 rounded border border-[var(--control-border)] px-1 text-[10px] text-[var(--muted)] hover:bg-[var(--control-hover)]"
              onClick={(event) => { event.stopPropagation(); data.onToggleCollapse?.(); }}
            >{data.collapsed ? '+' : '−'}</button>
          )}
        </div>
      </div>
      <Handle type="target" position={Position.Top} id="in" style={{ background: 'var(--graph-edge)' }} />
      <Handle type="source" position={Position.Right} id="result" style={{ background: 'var(--graph-edge-return)' }} />
      <Handle type="target" position={Position.Right} id="result" style={{ background: 'var(--graph-edge-yield)' }} />
      <Handle type="target" position={Position.Bottom} id="yield" style={{ background: 'var(--graph-edge-yield)' }} />
    </div>
  );
}

function ArithmeticNode({ data }: { data: any }) {
  const orderedBinary = data.orderedBinary;
  return (
    <div className="arithmetic-node" title={`MLIR: ${data.rawLabel || data.label}`}>
      <div className="arithmetic-node-shape">
        <span>{data.label}</span>
      </div>
      {orderedBinary ? (
        <>
          <Handle type="target" position={Position.Top} id="operand-0" style={{ left: '25%', background: 'var(--graph-edge)' }} />
          <Handle type="target" position={Position.Top} id="operand-1" style={{ left: '75%', background: 'var(--graph-edge)' }} />
        </>
      ) : (
        <Handle type="target" position={Position.Top} id="in" style={{ background: 'var(--graph-edge)' }} />
      )}
      <Handle type="source" position={Position.Bottom} id="result" style={{ background: 'var(--graph-edge-return)' }} />
    </div>
  );
}

const nodeTypes = { regionOp: RegionOpNode, arithmetic: ArithmeticNode };

const PYTORCH_BLOCK_ARG_MIN_WIDTH = 36;
const PYTORCH_BLOCK_ARG_HORIZONTAL_PADDING = 20;

function measureDisplayLabelWidth(label: string): number {
  if (typeof document === 'undefined') return label.length * 7;
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return label.length * 7;
  context.font = '11px "Geist Mono", ui-monospace, monospace';
  return context.measureText(label).width;
}

function pytorchBlockArgWidth(label: string): number {
  return Math.max(
    PYTORCH_BLOCK_ARG_MIN_WIDTH,
    Math.ceil(measureDisplayLabelWidth(label) + PYTORCH_BLOCK_ARG_HORIZONTAL_PADDING + 2),
  );
}

function numericNodeWidth(node: Node<any>, fallback = 100): number {
  // Display-mode styles are authoritative here; React Flow's measured width
  // can still reflect the previous MLIR label for one render.
  const width = node.style?.width ?? node.measured?.width ?? node.width;
  const numericWidth = typeof width === 'number' ? width : Number(width);
  return Number.isFinite(numericWidth) && numericWidth > 0 ? numericWidth : fallback;
}

function redistributePyTorchBlockArgs(displayedNodes: Node<any>[]): Node<any>[] {
  const byParent = new Map<string, Node<any>[]>();
  displayedNodes.forEach((node) => {
    if (!node.parentId || !node.data?.isBlockArg || (node.data.zone !== 'input' && node.data.zone !== 'output')) return;
    const children = byParent.get(node.parentId) ?? [];
    children.push(node);
    byParent.set(node.parentId, children);
  });

  return displayedNodes.map((node) => {
    const parentChildren = node.parentId ? byParent.get(node.parentId) : undefined;
    if (!parentChildren || (node.data?.zone !== 'input' && node.data?.zone !== 'output')) return node;
    const siblings = parentChildren.filter((child) => child.data.zone === node.data.zone);
    const parent = displayedNodes.find((candidate) => candidate.id === node.parentId);
    if (!parent) return node;
    const containerWidthValue = parent.measured?.width ?? parent.width ?? parent.style?.width;
    const containerWidth = typeof containerWidthValue === 'number' ? containerWidthValue : Number(containerWidthValue);
    if (!Number.isFinite(containerWidth) || containerWidth <= 0) return node;
    const widths = siblings.map((child) => numericNodeWidth(child));
    const padding = 18;
    const usableWidth = Math.max(0, containerWidth - padding * 2);
    const totalWidth = widths.reduce((sum, width) => sum + width, 0);
    const gap = Math.max(0, Math.min(16, (usableWidth - totalWidth) / Math.max(1, siblings.length - 1)));
    const occupiedWidth = totalWidth + gap * Math.max(0, siblings.length - 1);
    const index = siblings.findIndex((sibling) => sibling.id === node.id);
    const x = padding + Math.max(0, (usableWidth - occupiedWidth) / 2) + widths.slice(0, index).reduce((sum, width) => sum + width + gap, 0);
    return { ...node, position: { ...node.position, x: Math.min(Math.max(padding, x), Math.max(padding, containerWidth - padding - widths[index])) } };
  });
}

const PYTORCH_OPERATION_LABELS: Record<string, string> = {
  'tensor.empty': 'Empty Tensor',
  'linalg.fill': 'fillTensor()',
};

const ARITHMETIC_SYMBOLS: Record<string, string> = {
  'arith.addf': '+', 'arith.addi': '+', 'arith.subf': '-', 'arith.subi': '-',
  'arith.mulf': '*', 'arith.muli': '*', 'arith.divf': '/', 'arith.divsi': '/',
  'arith.divui': '/', 'math.powf': '^', 'math.powi': '^', 'powf': '^', 'powi': '^',
};

const DEFAULT_TORCH = `import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return torch.relu(y + b)

# You must define 'model' and 'inputs'
model = MyModel()
inputs = (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4))
`;
export default function App() {
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState<any | null>(null);
  const [codeContent, setCodeContent] = useState<string>('');
  const [pythonCode, setPythonCode] = useState<string>(DEFAULT_TORCH);
  const [activeTab, setActiveTab] = useState<'mlir' | 'python'>('python');
  const activeTabRef = useRef<'mlir' | 'python'>('python');

  const setTab = (tab: 'mlir' | 'python') => {
    activeTabRef.current = tab;
    setActiveTab(tab);
  };
  const [isCompiling, setIsCompiling] = useState(false);
  const [currentRunId, setCurrentRunId] = useState<string>('run_001');
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('http://localhost:8001');

  const [showPyTorchNames, setShowPyTorchNames] = useState<boolean>(false);
  const [runMetadataArgs, setRunMetadataArgs] = useState<string[]>([]);
  const [runSourceMeta, setRunSourceMeta] = useState<SourceMetadata>({});
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('llvm-explorer-theme') === 'dark');

  useEffect(() => {
    localStorage.setItem('llvm-explorer-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    return () => cancelAnimationFrame(frame);
  }, [isFullscreen]);

  useEffect(() => {
    async function loadRunData() {
      const ports = [8001, 8000];
      let data = null;
      let usedPort = 8001;

      for (const port of ports) {
        try {
          const res = await fetch(`http://localhost:${port}/api/runs/${currentRunId}`);
          if (res.ok) {
            data = await res.json();
            usedPort = port;
            break;
          }
        } catch {
          // try next port
        }
      }

      setApiBaseUrl(`http://localhost:${usedPort}`);

      if (data) {
        if (data.pytorch_args && Array.isArray(data.pytorch_args)) {
          setRunMetadataArgs(data.pytorch_args);
        } else {
          setRunMetadataArgs([]);
        }
        if (data.pytorch_source_meta && typeof data.pytorch_source_meta === 'object') {
          setRunSourceMeta(data.pytorch_source_meta);
        } else {
          setRunSourceMeta({});
        }
        const rawEvents = (data.events || data.passes || data.snapshots || (Array.isArray(data) ? data : [])) as any[];
        const parsedSnapshots = rawEvents
          .filter((e) => e.type === 'snapshot' || e.path)
          .map((e, idx) => ({
            id: e.id || `snap_${idx}`,
            name: e.name || `${e.pass || 'Pass'} (${e.kind || 'snapshot'})`,
            path: e.path,
            pass: e.pass,
            stage: e.stage || 'linalg',
          }));

        const items = parsedSnapshots.length > 0 ? parsedSnapshots : [];
        setSnapshots(items);
        if (items.length > 0) {
          setSelectedSnapshot(items[0]);
        } else {
          setSelectedSnapshot(null);
          setCodeContent('');
        }
      } else {
        // Fallback placeholder data if API fails to load data
        const mockSnapshots = [
          { id: '1', name: 'Wait for Data (no jsonl)', path: '' },
        ];
        setSnapshots(mockSnapshots);
        setSelectedSnapshot(null);
      }
    }

    // Clear stale state while loading new run
    setSelectedSnapshot(null);
    setCodeContent('');
    loadRunData();
  }, [currentRunId]);

  useEffect(() => {
    if (selectedSnapshot && selectedSnapshot.path) {
      // Derive the run id from the snapshot's own path (runs/<run_id>/stages/...)
      // so a newly compiled run never fetches files against a stale run id.
      const pathMatch = selectedSnapshot.path.match(/^runs\/([^/]+)\//);
      const snapshotRunId = pathMatch ? pathMatch[1] : currentRunId;

      fetch(`${apiBaseUrl}/api/runs/${snapshotRunId}/file?path=${encodeURIComponent(selectedSnapshot.path)}`)
        .then((res) => {
           if (!res.ok) throw new Error('Failed to fetch file');
           return res.json();
        })
        .then((data) => {
           setCodeContent(data.content || '');
        })
        .catch((err) => {
           console.error(err);
           setCodeContent(`// Could not load file from backend:\n// ${selectedSnapshot.path}`);
        });
    }
  }, [selectedSnapshot, apiBaseUrl]);

  const handleCompile = async () => {
    setIsCompiling(true);
    try {
      const res = await fetch(`${apiBaseUrl}/api/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: pythonCode })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Compilation failed');
      setCurrentRunId(data.id);
      if (data.pytorch_args && Array.isArray(data.pytorch_args)) {
        setRunMetadataArgs(data.pytorch_args);
      } else {
        setRunMetadataArgs([]);
      }
      if (data.pytorch_source_meta && typeof data.pytorch_source_meta === 'object') {
        setRunSourceMeta(data.pytorch_source_meta);
      } else {
        setRunSourceMeta({});
      }
      // Clear the stale snapshot so the old run's MLIR trace never renders
      // while the new run's timeline loads.
      setSelectedSnapshot(null);
      setCodeContent('');
      setTab('mlir');
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    } finally {
      setIsCompiling(false);
    }
  };

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<any>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const onConstrainedNodesChange = (changes: NodeChange<Node<any>>[]) => {
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const constrainedChanges = changes.map((change) => {
      if (change.type !== 'position' || !change.position) return change;
      const node = nodeById.get(change.id);
      if (!node?.parentId || (node.data?.zone !== 'input' && node.data?.zone !== 'output')) return change;
      const container = nodeById.get(node.parentId);
      if (!container) return change;

      const containerWidth = Number(container.measured?.width ?? container.width ?? container.style?.width ?? 0);
      const containerHeight = Number(container.measured?.height ?? container.height ?? container.style?.height ?? 0);
      const styledWidth = typeof node.style?.width === 'number' ? node.style.width : Number(node.style?.width);
      const styledHeight = typeof node.style?.height === 'number' ? node.style.height : Number(node.style?.height);
      const nodeWidth = node.measured?.width || node.width || (Number.isFinite(styledWidth) && styledWidth > 0 ? styledWidth : 100);
      const nodeHeight = node.measured?.height || node.height || (Number.isFinite(styledHeight) && styledHeight > 0 ? styledHeight : 40);
      const headerHeight = Number(container.data?.headerHeight ?? 37);
      const inputZoneHeight = Number(container.data?.inputZoneHeight ?? 58);
      const outputZoneHeight = Number(container.data?.outputZoneHeight ?? 58);
      const zoneTop = node.data.zone === 'input' ? headerHeight : containerHeight - outputZoneHeight;
      const zoneBottom = node.data.zone === 'input' ? headerHeight + inputZoneHeight : containerHeight;
      const maxX = Math.max(0, containerWidth - nodeWidth);
      const maxY = Math.max(zoneTop, zoneBottom - nodeHeight);

      return {
        ...change,
        position: {
          x: Math.min(maxX, Math.max(0, change.position.x)),
          y: Math.min(maxY, Math.max(zoneTop, change.position.y)),
        },
      };
    });
    onNodesChange(constrainedChanges);
  };

  const pytorchArgs = useMemo(() => {
    if (runMetadataArgs && runMetadataArgs.length > 0) return runMetadataArgs;
    return extractPyTorchArgsFromPython(pythonCode);
  }, [runMetadataArgs, pythonCode]);

  const { ssaMap, opLabelMap, compositeNodeIds } = useMemo(() => {
    const meta = Object.keys(runSourceMeta).length > 0 ? runSourceMeta : undefined;
    return buildPyTorchDisplayMaps(pythonCode, codeContent, nodes, meta);
  }, [pythonCode, codeContent, nodes, runSourceMeta]);

  const [collapsedCompositeIds, setCollapsedCompositeIds] = useState<Set<string>>(new Set());
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [tracedNodeId, setTracedNodeId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  useEffect(() => {
    const clearTraceOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setTracedNodeId(null);
        setContextMenu(null);
      }
    };
    document.addEventListener('keydown', clearTraceOnEscape);
    return () => document.removeEventListener('keydown', clearTraceOnEscape);
  }, []);

  useEffect(() => {
    setCollapsedCompositeIds(previous => {
      const next = new Set([...previous].filter(id => compositeNodeIds.includes(id)));
      compositeNodeIds.forEach(id => next.add(id));
      return next;
    });
  }, [compositeNodeIds.join('|')]);

  const toggleComposite = (id: string) => {
    setCollapsedCompositeIds(previous => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const displayNodes = useMemo(() => {
    const mappedNodes = nodes.map((node: Node<any>) => {
      // 1. Function Arguments (%arg0, %arg1, ...)
      if (node.type === 'input' || node.data?.isFuncArg) {
        const idx = node.data.argIndex ?? 0;
        const ptName = ssaMap[node.id] || pytorchArgs[idx];
        const rawLabel = String(node.data.rawLabel || node.id);
        const displayLabel = showPyTorchNames && ptName ? ptName : rawLabel;
        return {
          ...node,
          data: { ...node.data, rawLabel, label: displayLabel, pytorchName: ptName },
          style: { ...node.style, width: 'auto', minWidth: '36px', textAlign: 'center' as const },
        };
      }

      // 2. Region Block Arguments (%in, %in_0, %out, etc.)
      if (node.data?.isBlockArg) {
        const rawLabel = String(node.data.rawLabel || node.data.label);
        const operandName = typeof node.data.operandName === 'string' ? node.data.operandName : undefined;
        let inheritedName = operandName ? ssaMap[operandName] : undefined;
        if (!inheritedName && node.data.isOutputBlockArg) inheritedName = 'output';
        const displayLabel = showPyTorchNames && inheritedName ? inheritedName : rawLabel;
        return {
          ...node,
          data: { ...node.data, rawLabel, label: displayLabel, pytorchName: inheritedName },
          style: showPyTorchNames
            ? { ...node.style, width: pytorchBlockArgWidth(displayLabel), minWidth: PYTORCH_BLOCK_ARG_MIN_WIDTH, textAlign: 'center' as const, whiteSpace: 'nowrap' }
            : node.style,
        };
      }

      // Keep the MLIR spelling in rawLabel for tooltips and MLIR mode, while
      // using the concise operation name in PyTorch Names mode.
      if (node.data?.isYield) {
        const rawLabel = String(node.data.rawLabel || node.data.label);
        return { ...node, data: { ...node.data, rawLabel, label: showPyTorchNames ? 'yield' : rawLabel } };
      }

      // 3. Region container nodes (linalg.generic, etc.)
      if (node.type === 'regionOp') {
        const rawLabel = String(node.data.rawLabel || node.data.label);
        const ptLabel = opLabelMap[node.id];
        const displayLabel = showPyTorchNames && ptLabel ? ptLabel : rawLabel;
        const isComposite = compositeNodeIds.includes(node.id);
        return {
          ...node,
          data: {
            ...node.data,
            rawLabel,
            label: displayLabel,
            isComposite,
            collapsed: isComposite && collapsedCompositeIds.has(node.id),
            onToggleCollapse: () => toggleComposite(node.id),
          },
          style: isComposite && collapsedCompositeIds.has(node.id)
            ? { ...node.style, height: 116 }
            : node.style,
        };
      }

      // 4. Simple ops: arith ops → operator symbol; arith.constant → value; SSA result → name
      if (node.data?.rawOp) {
        const rawLabel = String(node.data.rawLabel || node.data.label);
        const arithmeticSymbol = ARITHMETIC_SYMBOLS[node.data.rawOp as string] || node.data.arithOp;
        const isArithmetic = Boolean(arithmeticSymbol);
        if (isArithmetic && showPyTorchNames) {
          return { ...node, type: 'arithmetic', data: { ...node.data, rawLabel, label: arithmeticSymbol } };
        }
        if (showPyTorchNames) {
          const semanticOperationLabel = PYTORCH_OPERATION_LABELS[node.data.rawOp as string];
          if (semanticOperationLabel) {
            return {
              ...node,
              data: { ...node.data, rawLabel, label: semanticOperationLabel },
              className: node.data.rawOp === 'tensor.empty' ? 'tensor-empty-node' : 'function-operation-node',
            };
          }
          const sourceLabel = opLabelMap[node.id];
          if (sourceLabel) {
            return { ...node, data: { ...node.data, rawLabel, label: sourceLabel } };
          }
          if (node.data.arithOp) {
            return { ...node, data: { ...node.data, rawLabel, label: node.data.arithOp } };
          }
          if (node.data.constValue != null) {
            return { ...node, data: { ...node.data, rawLabel, label: String(node.data.constValue) } };
          }
          const mlirResults: string[] = Array.isArray(node.data.mlirResults) ? node.data.mlirResults : [];
          const resultName = mlirResults.map((r: string) => ssaMap[r]).find(Boolean);
          if (resultName) {
            const shortOp = (node.data.rawOp as string).split('.').pop() ?? node.data.rawOp;
            return { ...node, data: { ...node.data, rawLabel, label: `${resultName} = ${shortOp}` } };
          }
        }
        return { ...node, type: 'default', data: { ...node.data, rawLabel, label: rawLabel } };
      }

      return node;
    });
    return showPyTorchNames ? redistributePyTorchBlockArgs(mappedNodes) : mappedNodes;
  }, [nodes, showPyTorchNames, pytorchArgs, ssaMap, opLabelMap, compositeNodeIds, collapsedCompositeIds]);

  const displayEdges = useMemo(() => {
    const collapsed = new Set(compositeNodeIds.filter(id => collapsedCompositeIds.has(id)));
    const parentById = new Map(nodes.map(node => [node.id, node.parentId]));
    const visibleOwner = (id: string) => {
      let current: string | undefined = id;
      while (current) {
        const parent = parentById.get(current) as string | undefined;
        if (parent && collapsed.has(parent)) return parent;
        current = parent;
      }
      return id;
    };
    return edges.map(edge => {
      const source = visibleOwner(edge.source);
      const target = visibleOwner(edge.target);
      const targetHandle = !showPyTorchNames && edge.targetHandle?.startsWith('operand-')
        ? undefined
        : edge.targetHandle;
      return { ...edge, source, target, targetHandle, hidden: source === target };
    });
  }, [edges, nodes, compositeNodeIds, collapsedCompositeIds, showPyTorchNames]);

  const visibleNodes = useMemo(() => {
    const collapsed = new Set(compositeNodeIds.filter(id => collapsedCompositeIds.has(id)));
    const parentById = new Map(nodes.map(node => [node.id, node.parentId]));
    return displayNodes.map(node => {
      let parent = node.parentId as string | undefined;
      let isHidden = false;
      while (parent) {
        if (collapsed.has(parent)) { isHidden = true; break; }
        parent = parentById.get(parent) as string | undefined;
      }
      return isHidden ? { ...node, hidden: true } : node;
    });
  }, [displayNodes, nodes, compositeNodeIds, collapsedCompositeIds]);

  useEffect(() => {
    const parsed = parseMLIRToGraph(codeContent);
    setNodes(parsed.nodes);
    setEdges(parsed.edges);
    setFocusedNodeId(null);
    setTracedNodeId(null);
    setContextMenu(null);
  }, [codeContent, setNodes, setEdges]);

  const focusSelection = useMemo(() => {
    const selectedIds = new Set<string>();
    const contextIds = new Set<string>();
    const emphasizedEdgeIds = new Set<string>();
    if (!focusedNodeId) return { selectedIds, contextIds, emphasizedEdgeIds };

    const childrenByParent = new Map<string, string[]>();
    const parentById = new Map(nodes.map((node) => [node.id, node.parentId]));
    nodes.forEach((node) => {
      if (!node.parentId) return;
      const children = childrenByParent.get(node.parentId) ?? [];
      children.push(node.id);
      childrenByParent.set(node.parentId, children);
    });

    const addDescendants = (id: string) => {
      if (selectedIds.has(id)) return;
      selectedIds.add(id);
      (childrenByParent.get(id) ?? []).forEach(addDescendants);
    };
    addDescendants(focusedNodeId);

    // Raw semantic edges are used here so collapsed display routing cannot
    // accidentally turn a sibling or unrelated node into a neighbor.
    edges.forEach((edge) => {
      const sourceSelected = selectedIds.has(edge.source);
      const targetSelected = selectedIds.has(edge.target);
      if (sourceSelected === targetSelected) {
        if (sourceSelected) emphasizedEdgeIds.add(edge.id);
        return;
      }
      const externalId = sourceSelected ? edge.target : edge.source;
      contextIds.add(externalId);
      emphasizedEdgeIds.add(edge.id);
    });

    // Keep hierarchy context visible without selecting any additional child.
    [...selectedIds, ...contextIds].forEach((id) => {
      let parent = parentById.get(id) as string | undefined;
      while (parent) {
        contextIds.add(parent);
        parent = parentById.get(parent) as string | undefined;
      }
    });

    return { selectedIds, contextIds, emphasizedEdgeIds };
  }, [focusedNodeId, nodes, edges]);

  const traceSelection = useMemo(() => {
    const tracedIds = new Set<string>();
    const tracedEdgeIds = new Set<string>();
    if (!tracedNodeId) return { tracedIds, tracedEdgeIds };

    const outgoing = new Map<string, Edge[]>();
    const childrenByParent = new Map<string, string[]>();
    edges.forEach((edge) => {
      const branch = outgoing.get(edge.source) ?? [];
      branch.push(edge);
      outgoing.set(edge.source, branch);
    });
    nodes.forEach((node) => {
      if (!node.parentId) return;
      const children = childrenByParent.get(node.parentId) ?? [];
      children.push(node.id);
      childrenByParent.set(node.parentId, children);
    });

    const visit = (id: string) => {
      if (tracedIds.has(id)) return;
      tracedIds.add(id);
      (outgoing.get(id) ?? []).forEach((edge) => {
        tracedEdgeIds.add(edge.id);
        visit(edge.target);
      });
    };
    const seedSubtree = (id: string) => {
      if (tracedIds.has(id)) return;
      tracedIds.add(id);
      (childrenByParent.get(id) ?? []).forEach(seedSubtree);
    };
    seedSubtree(tracedNodeId);
    [...tracedIds].forEach((id) => {
      (outgoing.get(id) ?? []).forEach((edge) => {
        tracedEdgeIds.add(edge.id);
        visit(edge.target);
      });
    });
    return { tracedIds, tracedEdgeIds };
  }, [tracedNodeId, nodes, edges]);

  const traceVisibleNodeIds = useMemo(() => {
    const visibleIds = new Set(traceSelection.tracedIds);
    if (!tracedNodeId) return visibleIds;
    const parentById = new Map(nodes.map((node) => [node.id, node.parentId]));
    traceSelection.tracedIds.forEach((id) => {
      let parent = parentById.get(id) as string | undefined;
      while (parent) {
        if (collapsedCompositeIds.has(parent)) visibleIds.add(parent);
        parent = parentById.get(parent) as string | undefined;
      }
    });
    return visibleIds;
  }, [tracedNodeId, traceSelection.tracedIds, nodes, collapsedCompositeIds]);

  const focusedNodes = useMemo(() => visibleNodes.map((node) => {
    if (!focusedNodeId && !tracedNodeId) return node;
    if (tracedNodeId) {
      const isTraced = traceVisibleNodeIds.has(node.id);
      return {
        ...node,
        style: {
          ...node.style,
          opacity: isTraced ? 1 : 0.18,
          outline: node.id === tracedNodeId ? '3px solid var(--graph-trace)' : undefined,
          outlineOffset: node.id === tracedNodeId ? '2px' : undefined,
          boxShadow: node.id === tracedNodeId ? '0 0 0 5px var(--graph-trace-ring), 0 8px 20px var(--graph-trace-shadow)' : node.style?.boxShadow,
          transition: 'opacity 180ms ease, box-shadow 180ms ease, outline 180ms ease',
        },
      };
    }
    const isFocused = focusSelection.selectedIds.has(node.id);
    const isContext = focusSelection.contextIds.has(node.id);
    return {
      ...node,
      style: {
        ...node.style,
        opacity: isFocused ? 1 : isContext ? 0.58 : 0.18,
        outline: node.id === focusedNodeId ? '3px solid var(--graph-focus)' : undefined,
        outlineOffset: node.id === focusedNodeId ? '2px' : undefined,
        boxShadow: node.id === focusedNodeId ? '0 0 0 5px var(--graph-focus-ring), 0 8px 20px var(--graph-focus-shadow)' : node.style?.boxShadow,
        transition: 'opacity 180ms ease, box-shadow 180ms ease, outline 180ms ease',
      },
    };
  }), [visibleNodes, focusedNodeId, tracedNodeId, traceVisibleNodeIds, focusSelection]);

  const focusedEdges = useMemo(() => displayEdges.map((edge) => {
    if (!focusedNodeId && !tracedNodeId) {
      return { ...edge, animated: false, className: `${edge.className ?? ''} graph-edge-flow`.trim() };
    }
    if (tracedNodeId) {
      const isTraced = traceSelection.tracedEdgeIds.has(edge.id);
      return {
        ...edge,
        style: {
          ...edge.style,
          opacity: isTraced ? 1 : 0.1,
          strokeWidth: isTraced ? Math.max(Number(edge.style?.strokeWidth ?? 1.5) + 1, 2.5) : edge.style?.strokeWidth,
          transition: 'opacity 180ms ease, stroke-width 180ms ease',
        },
        animated: isTraced,
        className: `${isTraced ? 'graph-edge-focused graph-edge-flow-active' : 'graph-edge-muted'} graph-edge-flow`,
      };
    }
    const isFocused = focusSelection.emphasizedEdgeIds.has(edge.id);
    return {
      ...edge,
      style: {
        ...edge.style,
        opacity: isFocused ? 1 : 0.1,
        strokeWidth: isFocused ? Math.max(Number(edge.style?.strokeWidth ?? 1.5) + 1, 2.5) : edge.style?.strokeWidth,
        transition: 'opacity 180ms ease, stroke-width 180ms ease',
      },
      animated: isFocused,
      className: `${isFocused ? 'graph-edge-focused graph-edge-flow-active' : 'graph-edge-muted'} graph-edge-flow`,
    };
  }), [displayEdges, focusedNodeId, tracedNodeId, traceSelection.tracedEdgeIds, focusSelection]);

  return (
    <div data-theme={isDarkMode ? 'dark' : 'light'} className="app-shell flex h-screen w-full bg-[var(--app-bg)] text-[var(--text)] overflow-hidden font-sans">
      {/* Sidebar / Timeline */}
      <div className={`${isFullscreen ? 'hidden' : 'flex'} w-[320px] border-r border-[var(--border)] bg-[var(--surface)] flex-col h-full shrink-0`}>
        <div className="p-5 border-b border-[var(--border)]">
          <h1 className="font-semibold text-[15px] flex items-center gap-2 text-[var(--text)]">
             <Activity className="w-[18px] h-[18px] text-[var(--muted)]" />
            AI Compiler Explorer
          </h1>
           <p className="text-[13px] text-[var(--muted)] mt-1">Optimization Timeline</p>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-1.5">
          {snapshots.length === 0 ? (
             <p className="text-[var(--muted)] text-sm italic">Loading passes...</p>
          ) : (
            snapshots.map((snap, idx) => {
              const isSelected = selectedSnapshot?.id === snap.id || selectedSnapshot?.path === snap.path;
              return (
                <button
                  key={snap.id || idx}
                  onClick={() => setSelectedSnapshot(snap)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all ${
                    isSelected
                       ? 'bg-[var(--control-active)] border-[var(--border)] shadow-sm'
                       : 'bg-[var(--surface)] border-transparent hover:bg-[var(--control-hover)] hover:border-[var(--border)]'
                  }`}
                >
                   <div className={`font-medium text-[13px] ${isSelected ? 'text-[var(--text)]' : 'text-[var(--muted-strong)]'}`}>
                    {snap.name || `Pass ${idx + 1}`}
                  </div>
                  {snap.path && (
                     <div className="text-[11px] text-[var(--muted)] mt-0.5 font-mono truncate">
                      {snap.path}
                    </div>
                  )}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Main Content Area */}
       <div className="flex-1 flex flex-col h-full bg-[var(--surface)]">
        {selectedSnapshot ? (
           <div className="flex flex-1 h-full overflow-hidden">
            {/* Code Editor */}
            <div className={`${isFullscreen ? 'hidden' : 'flex'} flex-1 flex-col border-r border-[var(--border)] h-full max-w-[50%]`}>
              <div className="px-4 py-2 bg-[var(--subtle-surface)] border-b border-[var(--border)] flex items-center justify-between">
                <div className="flex gap-2">
                  <button 
                    onClick={() => setTab('python')}
                     className={`px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors ${activeTab === 'python' ? 'bg-[var(--surface)] shadow-sm border border-[var(--border)] text-[var(--text)]' : 'text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--control-hover)]'}`}
                  >
                    <div className="flex items-center gap-1.5"><Code2 className="w-3.5 h-3.5" /> PyTorch Source</div>
                  </button>
                  <button 
                    onClick={() => setTab('mlir')}
                     className={`px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors ${activeTab === 'mlir' ? 'bg-[var(--surface)] shadow-sm border border-[var(--border)] text-[var(--text)]' : 'text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--control-hover)]'}`}
                  >
                    <div className="flex items-center gap-1.5"><FileCode className="w-3.5 h-3.5" /> MLIR Trace</div>
                  </button>
                </div>
                
                {activeTab === 'python' && (
                  <button 
                    onClick={handleCompile}
                    disabled={isCompiling}
                     className="flex items-center gap-1.5 bg-[var(--button-bg)] hover:bg-[var(--button-hover)] text-[var(--button-text)] px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5" />
                    {isCompiling ? 'Compiling...' : 'Compile'}
                  </button>
                )}
              </div>
               <div className="flex-1 bg-[var(--editor-bg)]">
                <Editor
                  height="100%"
                  language={activeTab === 'python' ? 'python' : 'llvm'}
                   theme={isDarkMode ? 'vs-dark' : 'light'}
                  path={activeTab === 'python' ? 'source.py' : (selectedSnapshot?.path || 'trace.mlir')}
                  value={activeTab === 'python' ? pythonCode : codeContent}
                  onChange={(val) => {
                    if (activeTabRef.current === 'python') {
                      setPythonCode(val || '');
                    }
                  }}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    fontFamily: 'var(--font-mono)',
                    lineHeight: 1.6,
                    readOnly: activeTab === 'mlir',
                    padding: { top: 24, bottom: 24 },
                    scrollBeyondLastLine: false,
                    renderLineHighlight: activeTab === 'python' ? 'line' : 'none',
                    hideCursorInOverviewRuler: activeTab === 'mlir',
                  }}
                />
              </div>
            </div>

            {/* Graph View */}
            <div className={`flex flex-col bg-[var(--graph-bg)] relative ${isFullscreen ? 'fixed inset-0 z-50 w-screen h-screen' : 'flex-1 h-full'}`}>
              <div className="px-4 py-2.5 border-b border-[var(--border)] flex items-center justify-between bg-[var(--surface-translucent)] backdrop-blur-sm sticky top-0 z-10">
                <div className="flex items-center gap-2">
                   {isFullscreen ? <Minimize2 className="w-4 h-4 text-[var(--muted)]" /> : <Maximize2 className="w-4 h-4 text-[var(--muted)]" />}
                   <span className="text-[13px] font-medium text-[var(--muted-strong)]">Control Flow Graph</span>
                </div>
                <div className="flex items-center gap-3">
                  <label
                    title="Show original PyTorch forward parameter names alongside MLIR argument names."
                     className="flex items-center gap-2 text-xs text-[var(--muted-strong)] font-medium cursor-pointer select-none"
                  >
                    <input
                      type="checkbox"
                      checked={showPyTorchNames}
                      onChange={(e) => setShowPyTorchNames(e.target.checked)}
                       className="rounded border-[var(--control-border)] text-[var(--button-bg)] focus:ring-[var(--button-bg)] h-3.5 w-3.5"
                    />
                    Show PyTorch Names
                  </label>
                   <button onClick={() => setIsDarkMode((dark) => !dark)} aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`} title={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`} className="p-1.5 rounded-md text-[var(--muted-strong)] hover:bg-[var(--control-hover)] transition-colors">
                     {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                   </button>
                   <button onClick={() => setIsFullscreen((fullscreen) => !fullscreen)} className="text-xs px-2 py-1 rounded bg-[var(--control-bg)] hover:bg-[var(--control-hover)] text-[var(--muted-strong)] font-medium transition-colors">
                     {isFullscreen ? 'Exit graph fullscreen' : 'Fullscreen graph'}
                   </button>
                </div>
              </div>
              <div className="flex-1 w-full h-full">
                <ReactFlow 
        nodes={focusedNodes}
        edges={focusedEdges}
                   onNodesChange={onConstrainedNodesChange}
                   onEdgesChange={onEdgesChange}
                  onNodeClick={(_, node) => {
                    setContextMenu(null);
                    setFocusedNodeId((current) => current === node.id ? null : node.id);
                  }}
                  onNodeContextMenu={(event, node) => {
                    event.preventDefault();
                    setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
                  }}
                  onPaneClick={() => {
                    setFocusedNodeId(null);
                    setTracedNodeId(null);
                    setContextMenu(null);
                  }}
                  nodeTypes={nodeTypes} 
                  fitView
                >
                   <Background color="var(--graph-grid)" gap={16} />
                   <Controls className="graph-controls !border-[var(--border)] !shadow-sm" showInteractive={false} />
                </ReactFlow>
                {contextMenu && (
                  <div
                    className="fixed z-[100] min-w-28 rounded-md border border-[var(--border)] bg-[var(--surface)] p-1 shadow-lg"
                    style={{ left: contextMenu.x, top: contextMenu.y }}
                    onClick={(event) => event.stopPropagation()}
                  >
                    <button
                      type="button"
                      className="w-full rounded px-3 py-1.5 text-left text-xs font-medium text-[var(--text)] hover:bg-[var(--control-hover)]"
                      onClick={() => {
                        setTracedNodeId(contextMenu.nodeId);
                        setFocusedNodeId(null);
                        setContextMenu(null);
                      }}
                    >
                      Trace
                    </button>
                  </div>
                )}
                {nodes.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-[var(--muted)] text-sm">
                     No graph topology parsing available for this code snippet
                   </div>
                )}
              </div>
            </div>
          </div>
        ) : (
           <div className="flex-1 flex flex-col items-center justify-center text-[var(--muted)]">
             <Activity className="w-8 h-8 mb-3 text-[var(--muted)]" />
            <p className="text-[14px]">Select a pass from the timeline</p>
          </div>
        )}
      </div>
    </div>
  );
}
