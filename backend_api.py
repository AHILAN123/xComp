from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import uuid
import subprocess
import re
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(title="AI Compiler Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = Path("runs")

class CompileRequest(BaseModel):
    code: str

@app.options("/api/compile")
def compile_options():
    return {"status": "ok"}

@app.post("/api/compile")
def compile_code(req: CompileRequest):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = run_dir / "stages" / "02_linalg"
    stages_dir.mkdir(parents=True, exist_ok=True)
    
    wrapper_code = f"""import os
import sys
import json
import inspect
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
{req.code}
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
    with open("{str(run_dir.resolve())}/pytorch_args.json", "w") as f:
        json.dump(pytorch_args, f)
except Exception:
    pass

try:
    mlir_module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    mlir_module.context.enable_multithreading(False)
    pm = PassManager.parse("builtin.module(func.func(canonicalize,cse))", context=mlir_module.context)
    pm.enable_ir_printing()
    
    sys.stderr = open("{str(run_dir.resolve())}/trace.log", "w")
    pm.run(mlir_module.operation)
    sys.stderr.close()
except Exception as e:
    sys.stderr = sys.__stderr__
    print(f"Compilation error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    wrapper_path = run_dir / "wrapper.py"
    wrapper_path.write_text(wrapper_code)

    env = os.environ.copy()
    python_exe = sys.executable
    res = subprocess.run([python_exe, str(wrapper_path)], capture_output=True, text=True, env=env)
    
    if res.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Compilation Failed:\n{res.stderr}\n{res.stdout}")

    # MLIR's IR dumps are emitted at the C++ level to file descriptor 2, which
    # bypasses any Python-level sys.stderr swap inside the wrapper. The
    # subprocess pipe above DID capture them, so persist them to trace.log.
    trace_log = run_dir / "trace.log"
    trace_log.write_text(res.stderr or "")

    events = []
    events.append({"type": "run_start", "run_id": run_id})
    current_pass = None
    current_ir = []
    ir_count = 0
    
    trace_log = run_dir / "trace.log"
    if trace_log.exists():
        with open(trace_log, "r") as f:
            for line in f:
                match = re.match(r"// -----// IR Dump (Before|After) (.*?) \((.*?)\)", line.strip())
                if match:
                    if current_pass and current_ir:
                        filename = f"{ir_count:02d}_before_{current_pass}.mlir"
                        filepath = stages_dir / filename
                        filepath.write_text("".join(current_ir))
                        events.append({"type": "snapshot", "kind": "before", "pass": current_pass, "path": f"runs/{run_id}/stages/02_linalg/{filename}"})
                        ir_count += 1
                        current_ir = []
                    kind = match.group(1).lower()
                    pass_name = match.group(3)
                    current_pass = pass_name
                    events.append({"type": "pass_start", "pass": pass_name, "stage": "linalg"})
                else:
                    if current_pass is not None:
                        current_ir.append(line)
        
        if current_pass and current_ir:
            filename = f"{ir_count:02d}_before_{current_pass}.mlir"
            filepath = stages_dir / filename
            filepath.write_text("".join(current_ir))
            events.append({"type": "snapshot", "kind": "before", "pass": current_pass, "path": f"runs/{run_id}/stages/02_linalg/{filename}"})
            events.append({"type": "pass_end", "pass": current_pass})
            
    events.append({"type": "run_end", "run_id": run_id})
    
    events_file = run_dir / "events.jsonl"
    with open(events_file, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
            
    pytorch_args_file = run_dir / "pytorch_args.json"
    pytorch_args = []
    if pytorch_args_file.exists():
        try:
            with open(pytorch_args_file, "r") as f:
                pytorch_args = json.load(f)
        except Exception:
            pass

    return {"id": run_id, "pytorch_args": pytorch_args}

@app.get("/api/runs")
def get_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir():
            runs.append({"id": run_dir.name})
    return runs

@app.get("/api/runs/{run_id}")
def get_run_details(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    events_file = run_dir / "events.jsonl"
    if not events_file.exists():
        raise HTTPException(status_code=404, detail="events.jsonl not found")
        
    events = []
    with open(events_file, "r") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
                
    pytorch_args_file = run_dir / "pytorch_args.json"
    pytorch_args = []
    if pytorch_args_file.exists():
        try:
            with open(pytorch_args_file, "r") as f:
                pytorch_args = json.load(f)
        except Exception:
            pass

    return {"id": run_id, "events": events, "pytorch_args": pytorch_args}

@app.get("/api/runs/{run_id}/file")
def get_run_file(run_id: str, path: str):
    run_dir = (RUNS_DIR / run_id).resolve()
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
        
    try:
        rel_path = path
        
        # When compiling dynamically, snapshots map to 'runs/run_XXX...' but the
        # frontend path requests pass `runs/run_XXX...`
        # We need to strip ANY of that prefix safely.
        import urllib.parse
        rel_path = urllib.parse.unquote(path)
        
        parts = rel_path.split("/")
        
        # If the path already includes run_id, strip runs/<run_id>
        if len(parts) > 2 and parts[0] == "runs" and parts[1] == run_id:
            rel_path = "/".join(parts[2:])
        # Or if it has 'stages', find that
        elif 'stages' in parts:
            idx = parts.index('stages')
            rel_path = "/".join(parts[idx:])
            
        requested_path = (run_dir / rel_path).resolve()
        
        # Security: ensure path is within run_dir
        if not str(requested_path).startswith(str(run_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")
        
    print(f"DEBUG file lookup: {requested_path}, exists={requested_path.exists()}, is_file={requested_path.is_file()}")

    if not requested_path.exists() or not requested_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path} (resolved to {requested_path})")
        
    with open(requested_path, "r") as f:
        content = f.read()
        
    return {"content": content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
