from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import uuid
import subprocess
import re
import ast
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


def extract_source_metadata(code: str) -> dict:
    """Create source annotations without changing or inspecting the MLIR."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"args": [], "expressions": []}

    forward = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    if forward is None:
        return {"args": [], "expressions": []}

    args = [a.arg for a in forward.args.args if a.arg not in {"self", "cls"}]
    expressions = []
    next_id = 0
    bindings = set(args)

    binary = {
        ast.Add: ("add", "+"), ast.Sub: ("sub", "-"), ast.Mult: ("mul", "*"),
        ast.Div: ("div", "/"), ast.MatMult: ("matmul", "@"), ast.Pow: ("pow", "**"),
    }
    call_symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/", "matmul": "@", "mm": "@", "bmm": "@"}

    def visit_expr(node):
        nonlocal next_id
        if isinstance(node, ast.Name):
            return node.id, node.id, None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = str(node.value)
            return value, value, None
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left = visit_expr(node.left)
            right = visit_expr(node.right)
            op, symbol = binary[type(node.op)]
            display = f"{left[1]} {symbol} {right[1]}"
            value_key = display
            expressions.append({"id": f"expr_{next_id}", "op": op, "display": display,
                                "valueName": value_key, "inputs": [left[0], right[0]]})
            next_id += 1
            return value_key, display, op
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = visit_expr(node.operand)
            display = f"-{inner[1]}"
            expressions.append({"id": f"expr_{next_id}", "op": "neg", "display": display,
                                "valueName": display, "inputs": [inner[0]]})
            next_id += 1
            return display, display, "neg"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = "call"
            values = [visit_expr(arg) for arg in node.args]
            symbol = call_symbols.get(name)
            display = f"{values[0][1]} {symbol} {values[1][1]}" if symbol and len(values) >= 2 else f"{name}({', '.join(v[1] for v in values)})"
            expressions.append({"id": f"expr_{next_id}", "op": name, "display": display,
                                "valueName": display, "inputs": [v[0] for v in values]})
            next_id += 1
            return display, display, name
        return ast.unparse(node), ast.unparse(node), None

    for statement in forward.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            expression_start = len(expressions)
            value_key, display, op = visit_expr(statement.value)
            if len(expressions) > expression_start:
                # The root operation already contains the real source inputs.
                # Promote it to the Python assignment's SSA-level value instead
                # of wrapping it in a synthetic operation.
                root = expressions[-1]
                root["valueName"] = target
                root["assignedName"] = target
            else:
                expressions.append({"id": f"expr_{next_id}", "op": op or "value", "display": display,
                                    "valueName": target, "assignedName": target, "inputs": [value_key]})
                next_id += 1
            bindings.add(target)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            visit_expr(statement.value)
    return {"args": args, "expressions": expressions}

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

    source_metadata = extract_source_metadata(req.code)
    (run_dir / "pytorch_source_meta.json").write_text(json.dumps(source_metadata))
    
    # Serialize user code as a safe Python string literal for use inside the wrapper
    user_code_repr = repr(req.code)
    
    wrapper_code = f"""import os
import sys
import json
import inspect
import ast
import textwrap
import torch
import torch_mlir
from torch_mlir.passmanager import PassManager

# -- User Code --
{req.code}
# ---------------

try:
    pytorch_args = []
    pytorch_source_meta = {{"args": [], "assignments": [], "ops": []}}
    if 'model' in locals():
        m = locals()['model']
        sig = None
        if hasattr(m, 'forward') and callable(m.forward):
            sig = inspect.signature(m.forward)
        elif callable(m):
            sig = inspect.signature(m)
        if sig:
            for p in sig.parameters.values():
                if p.name not in ('self', 'cls'):
                    pytorch_args.append(p.name)
                    pytorch_source_meta["args"].append(p.name)

    # Parse source assignments and operations from user code
    user_code = textwrap.dedent({user_code_repr})
    try:
        tree = ast.parse(user_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'forward':
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        lhs_names = []
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                lhs_names.append(t.id)
                        if lhs_names:
                            rhs_src = ast.unparse(stmt.value)
                            rhs_idents = [n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)]
                            pytorch_source_meta["assignments"].append({{
                                "lhs": lhs_names[0],
                                "rhs": rhs_src,
                                "rhs_idents": list(dict.fromkeys(rhs_idents))
                            }})
                    if isinstance(stmt, ast.Return) and stmt.value:
                        pytorch_source_meta["ops"].append({{
                            "kind": "return",
                            "expr": ast.unparse(stmt.value)
                        }})
    except Exception:
        pass

    with open("{str(run_dir.resolve())}/pytorch_args.json", "w") as f:
        json.dump(pytorch_args, f)
    with open("{str(run_dir.resolve())}/pytorch_source_meta_runtime.json", "w") as f:
        json.dump(pytorch_source_meta, f)
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

    pytorch_source_meta_file = run_dir / "pytorch_source_meta.json"
    pytorch_source_meta = {}
    if pytorch_source_meta_file.exists():
        try:
            with open(pytorch_source_meta_file, "r") as f:
                pytorch_source_meta = json.load(f)
        except Exception:
            pass

    return {"id": run_id, "pytorch_args": pytorch_args, "pytorch_source_meta": pytorch_source_meta}

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

    pytorch_source_meta_file = run_dir / "pytorch_source_meta.json"
    pytorch_source_meta = {}
    if pytorch_source_meta_file.exists():
        try:
            with open(pytorch_source_meta_file, "r") as f:
                pytorch_source_meta = json.load(f)
        except Exception:
            pass

    return {"id": run_id, "events": events, "pytorch_args": pytorch_args, "pytorch_source_meta": pytorch_source_meta}

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
