import { useEffect, useState, useMemo, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { ReactFlow, Background, Controls, Handle, Position, useNodesState, useEdgesState } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { FileCode, Activity, Maximize2, Play, Code2 } from 'lucide-react';
import { parseMLIRToGraph, extractPyTorchArgsFromPython, buildSSADisplayNameMap } from './mlirParser';

function RegionOpNode({ data }: { data: any }) {
  return (
    <div className="h-full w-full rounded-xl border border-zinc-300 bg-zinc-50/50 shadow-sm overflow-hidden">
      <div className="px-3 py-2 bg-white/90 border-b border-zinc-200">
        <span className="text-[11px] font-mono text-zinc-800 truncate block">{data.label}</span>
      </div>
      <Handle type="target" position={Position.Top} id="in" style={{ background: '#71717a' }} />
      <Handle type="source" position={Position.Right} id="result" style={{ background: '#18181b' }} />
      <Handle type="target" position={Position.Right} id="result" style={{ background: '#f59e0b' }} />
      <Handle type="target" position={Position.Bottom} id="yield" style={{ background: '#f59e0b' }} />
    </div>
  );
}

const nodeTypes = { regionOp: RegionOpNode };

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
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  };

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

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const pytorchArgs = useMemo(() => {
    if (runMetadataArgs && runMetadataArgs.length > 0) {
      return runMetadataArgs;
    }
    return extractPyTorchArgsFromPython(pythonCode);
  }, [runMetadataArgs, pythonCode]);

  const ssaMap = useMemo(() => {
    return buildSSADisplayNameMap(pythonCode, codeContent, nodes);
  }, [pythonCode, codeContent, nodes]);

  const displayNodes = useMemo(() => {
    return nodes.map((node) => {
      // 1. Function Arguments (%arg0, %arg1, ...)
      if (node.type === 'input' || node.data?.isFuncArg) {
        const idx = node.data.argIndex ?? 0;
        const ptName = ssaMap[node.id] || pytorchArgs[idx];
        const rawLabel = node.data.rawLabel || node.id;
        const displayLabel = showPyTorchNames && ptName ? ptName : rawLabel;

        return {
          ...node,
          data: {
            ...node.data,
            rawLabel,
            label: displayLabel,
            pytorchName: ptName,
          },
          style: {
            ...node.style,
            width: 'auto',
            minWidth: '36px',
            textAlign: 'center',
          },
        };
      }

      // 2. Region Block Arguments (%in, %in_0, %out, etc.)
      if (node.data?.isBlockArg) {
        const rawLabel = node.data.rawLabel || node.data.label;
        const operandName = node.data.operandName;
        let inheritedName = operandName ? ssaMap[operandName] : undefined;
        if (!inheritedName && node.data.isOutputBlockArg) {
          inheritedName = 'output';
        }
        const displayLabel = showPyTorchNames && inheritedName ? inheritedName : rawLabel;

        return {
          ...node,
          data: {
            ...node.data,
            rawLabel,
            label: displayLabel,
            pytorchName: inheritedName,
          },
        };
      }

      return node;
    });
  }, [nodes, showPyTorchNames, pytorchArgs, ssaMap]);

  useEffect(() => {
    const parsed = parseMLIRToGraph(codeContent);
    setNodes(parsed.nodes);
    setEdges(parsed.edges);
  }, [codeContent, setNodes, setEdges]);

  return (
    <div className="flex h-screen w-full bg-zinc-50 text-zinc-900 overflow-hidden font-sans">
      {/* Sidebar / Timeline */}
      <div className="w-[320px] border-r border-zinc-200 bg-white flex flex-col h-full shrink-0">
        <div className="p-5 border-b border-zinc-100">
          <h1 className="font-semibold text-[15px] flex items-center gap-2 text-zinc-900">
            <Activity className="w-[18px] h-[18px] text-zinc-400" />
            AI Compiler Explorer
          </h1>
          <p className="text-[13px] text-zinc-500 mt-1">Optimization Timeline</p>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-1.5">
          {snapshots.length === 0 ? (
            <p className="text-zinc-400 text-sm italic">Loading passes...</p>
          ) : (
            snapshots.map((snap, idx) => {
              const isSelected = selectedSnapshot?.id === snap.id || selectedSnapshot?.path === snap.path;
              return (
                <button
                  key={snap.id || idx}
                  onClick={() => setSelectedSnapshot(snap)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all ${
                    isSelected
                      ? 'bg-zinc-100 border-zinc-200 shadow-sm'
                      : 'bg-white border-transparent hover:bg-zinc-50 hover:border-zinc-200'
                  }`}
                >
                  <div className={`font-medium text-[13px] ${isSelected ? 'text-zinc-900' : 'text-zinc-700'}`}>
                    {snap.name || `Pass ${idx + 1}`}
                  </div>
                  {snap.path && (
                    <div className="text-[11px] text-zinc-500 mt-0.5 font-mono truncate">
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
      <div className="flex-1 flex flex-col h-full bg-white">
        {selectedSnapshot ? (
          <div className="flex flex-1 h-full overflow-hidden">
            {/* Code Editor */}
            <div className="flex-1 flex flex-col border-r border-zinc-200 h-full max-w-[50%]">
              <div className="px-4 py-2 bg-zinc-50 border-b border-zinc-200 flex items-center justify-between">
                <div className="flex gap-2">
                  <button 
                    onClick={() => setTab('python')}
                    className={`px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors ${activeTab === 'python' ? 'bg-white shadow-sm border border-zinc-200 text-zinc-900' : 'text-zinc-500 hover:text-zinc-700 hover:bg-zinc-100'}`}
                  >
                    <div className="flex items-center gap-1.5"><Code2 className="w-3.5 h-3.5" /> PyTorch Source</div>
                  </button>
                  <button 
                    onClick={() => setTab('mlir')}
                    className={`px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors ${activeTab === 'mlir' ? 'bg-white shadow-sm border border-zinc-200 text-zinc-900' : 'text-zinc-500 hover:text-zinc-700 hover:bg-zinc-100'}`}
                  >
                    <div className="flex items-center gap-1.5"><FileCode className="w-3.5 h-3.5" /> MLIR Trace</div>
                  </button>
                </div>
                
                {activeTab === 'python' && (
                  <button 
                    onClick={handleCompile}
                    disabled={isCompiling}
                    className="flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 text-white px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5" />
                    {isCompiling ? 'Compiling...' : 'Compile'}
                  </button>
                )}
              </div>
              <div className="flex-1 bg-white">
                <Editor
                  height="100%"
                  language={activeTab === 'python' ? 'python' : 'llvm'}
                  theme="light"
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
            <div className={`flex flex-col bg-zinc-50/50 relative ${isFullscreen ? 'fixed inset-0 z-50 w-screen h-screen' : 'flex-1 h-full'}`}>
              <div className="px-4 py-2.5 border-b border-zinc-200 flex items-center justify-between bg-white/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="flex items-center gap-2">
                  <Maximize2 className="w-4 h-4 text-zinc-400" />
                  <span className="text-[13px] font-medium text-zinc-600">Control Flow Graph</span>
                </div>
                <div className="flex items-center gap-3">
                  <label
                    title="Show original PyTorch forward parameter names alongside MLIR argument names."
                    className="flex items-center gap-2 text-xs text-zinc-600 font-medium cursor-pointer select-none"
                  >
                    <input
                      type="checkbox"
                      checked={showPyTorchNames}
                      onChange={(e) => setShowPyTorchNames(e.target.checked)}
                      className="rounded border-zinc-300 text-zinc-900 focus:ring-zinc-500 h-3.5 w-3.5"
                    />
                    Show PyTorch Names
                  </label>
                  <button
                    onClick={toggleFullscreen}
                    className="text-xs px-2 py-1 rounded bg-zinc-100 hover:bg-zinc-200 text-zinc-600 font-medium transition-colors"
                  >
                    {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                  </button>
                </div>
              </div>
              <div className="flex-1 w-full h-full">
                <ReactFlow 
                  nodes={displayNodes} 
                  edges={edges} 
                  onNodesChange={onNodesChange}
                  onEdgesChange={onEdgesChange}
                  nodeTypes={nodeTypes} 
                  fitView
                >
                  <Background color="#e4e4e7" gap={16} />
                  <Controls className="!border-zinc-200 !shadow-sm" showInteractive={false} />
                </ReactFlow>
                {nodes.length === 0 && (
                   <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-zinc-400 text-sm">
                     No graph topology parsing available for this code snippet
                   </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-zinc-400">
            <Activity className="w-8 h-8 mb-3 text-zinc-300" />
            <p className="text-[14px]">Select a pass from the timeline</p>
          </div>
        )}
      </div>
    </div>
  );
}
