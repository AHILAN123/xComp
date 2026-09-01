# PLAN.md — AI Compiler Optimization Explorer Plan

This document outlines the architecture, data models, pipelines, milestones, and development strategy for the AI Compiler Optimization Explorer.

---

## 1. Current Environment

The current development environment is structured as follows:

| Component | Status | Version / Path | Note |
|---|---|---|---|
| **OS** | Linux (Arch Linux) | N/A | Local packaging via pacman/AUR |
| **Shell** | Zsh | `/usr/bin/zsh` | Default terminal shell |
| **Python** | Installed | `3.14.7` (System) | Note: Too new for pre-built PyTorch/torch-mlir wheels |
| **Clang** | Installed | `22.1.8` at `/usr/bin/clang` | Modern C/C++ front-end available |
| **LLVM (CLI)** | Missing | N/A | No `llc`, `opt`, `llvm-as`, etc. |
| **MLIR (CLI)** | Missing | N/A | No `mlir-opt`, `mlir-translate`, `mlir-cpu-runner` |
| **PyTorch** | Missing | N/A | No `torch` Python package |
| **torch-mlir** | Missing | N/A | No `torch_mlir` Python package |
| **stablehlo** | Missing | N/A | No `stablehlo` Python package |
| **Conda** | Missing | N/A | Conda package manager not available |
| **Project Dir** | Empty | `/home/tielixir/Coding/Projects/LLVMOptimizer` | Fresh workspace |

---

## 2. Available Compiler Components & Installation Plan

To build our explorer, we need compiler toolchains capable of emitting pass trace information and IR dumps.

### Required Packages (Arch Linux)
Since we are on Arch Linux, we can install the official packages for LLVM and MLIR:
* **LLVM package** (provides `opt`, `llc`, `llvm-dis`):
  `sudo pacman -S llvm`
* **MLIR package** (provides `mlir-opt`, `mlir-translate`):
  `sudo pacman -S mlir`

### Python Ecosystem Setup
System Python is 3.14.7. PyTorch and torch-mlir packages are not fully compatible with Python 3.14 yet.
* **Mitigation:** Install Python 3.12 (available via AUR as `python312` or using `pyenv`) to run the PyTorch and torch-mlir pipeline.
* Create a virtual environment with Python 3.12:
  `python3.12 -m venv venv`
  `source venv/bin/activate`
* Install dependencies:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  `pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir/releases`

---

## 3. Candidate Pipelines

We consider three candidate compilation pipelines:

### Pipeline A: LLVM IR Optimization Pipeline (Highly Feasible / immediate bootstrap)
```text
C/C++ Source ──► Clang (LLVM IR) ──► opt (Passes) ──► llc (Target assembly)
```
* **Pros:** Fast to set up. Requires only system `llvm` package. `opt` provides very stable `-print-before-all` / `-print-after-all` outputs.
* **Cons:** Low-level; does not represent high-level tensor operations or PyTorch models directly.

### Pipeline B: Linalg-based MLIR Pipeline (MVP Chosen Target)
```text
PyTorch Model ──► Torch FX ──► torch-mlir (Torch Dialect) ──► Linalg Dialect ──► LLVM Dialect ──► LLVM IR
```
* **Pros:** Represents true machine learning model compilation. Captures lowering from high-level tensors to loop nests and vector instructions.
* **Cons:** Setup is more complex due to Python 3.12 venv and torch-mlir nightly dependencies.

### Pipeline C: StableHLO-based MLIR Pipeline (Post-MVP Target)
```text
PyTorch Model ──► StableHLO ──► Linalg ──► LLVM Dialect ──► LLVM IR
```
* **Pros:** StableHLO represents a standard portable representation used by XLA/JAX. Excellent for benchmarking.
* **Cons:** High complexity in torch-mlir StableHLO export paths.

---

## 4. Chosen MVP Pipeline

We will design a core architecture capable of ingestion from **both Pipeline A (LLVM IR)** and **Pipeline B (torch-mlir / Linalg)**.
* **Bootstrapping Phase:** Prove the capture format and normalizer logic using **Pipeline A** first (as it requires no PyTorch/Python 3.12 installation and can be tested immediately).
* **MVP Target Demo:** Run and visualize a tiny PyTorch MLP compilation under **Pipeline B**.

---

## 5. Architecture

The architecture consists of the following decoupled modules:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPILATION TOOLCHAIN                           │
│                                                                        │
│ PyTorch (FX) ──► torch-mlir ──► mlir-opt (Linalg/SCF) ──► llc (LLVM)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (Instruments: -mlir-print-ir-before/after-all)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           TRACE COLLECTOR                              │
│                                                                        │
│ Runs compilation, hooks outputs, extracts IR snapshots + pass events.  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼ (JSONL file + raw IR snapshots directory)
┌────────────────────────────────────────────────────────────────────────┐
│                        NORMALIZER & IR PARSER                          │
│                                                                        │
│ Parses textual MLIR/LLVM IR into structured dialect/instruction nodes. │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           PROVENANCE ENGINE                            │
│                                                                        │
│ Resolves cross-stage mapping (same location, structural, dataflow).     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼ (SQLite Database)
┌────────────────────────────────────────────────────────────────────────┐
│                         BACKEND API (FastAPI)                          │
│                                                                        │
│ Serves runs, stages, operations, passes, diffs, and provenance graphs. │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND WEB EXPLORER                          │
│                                                                        │
│ Timeline UI + CodeMirror Diff + React Flow Lineage Graph + Inspector.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Model

### CompilationRun
A single run representing a compilation event.
```json
{
  "run_id": "run_20260830_mlp",
  "model_name": "tiny_mlp",
  "pipeline_type": "torch-mlir",
  "environment": {
    "python": "3.12.8",
    "torch": "2.4.0",
    "torch_mlir": "20260815.123",
    "mlir": "22.1.8"
  },
  "created_at": "2026-08-30T12:50:00Z"
}
```

### Stage
A specific dialetical or IR representation layer.
```json
{
  "stage_id": "stage_linalg",
  "run_id": "run_20260830_mlp",
  "name": "Linalg Dialect",
  "order_index": 2,
  "ir_format": "mlir"
}
```

### PassEvent
An optimization pass execution details.
```json
{
  "pass_id": "pass_canonicalize_1",
  "run_id": "run_20260830_mlp",
  "name": "canonicalize",
  "stage_id": "stage_linalg",
  "pipeline_position": 4,
  "duration_ms": 12.4,
  "before_ir_path": "stages/02_linalg/before_canonicalize.mlir",
  "after_ir_path": "stages/02_linalg/after_canonicalize.mlir"
}
```

### Operation
An individual instruction/op node.
```json
{
  "op_id": "op_linalg_matmul_04",
  "run_id": "run_20260830_mlp",
  "stage_id": "stage_linalg",
  "pass_id": "pass_canonicalize_1",
  "name": "linalg.matmul",
  "dialect": "linalg",
  "location": "loc(\"model.py\":42:15)",
  "raw_text": "%5 = linalg.matmul ins(%1, %2) outs(%4)",
  "operands": ["%1", "%2", "%4"],
  "results": ["%5"],
  "attributes": {}
}
```

### Relation (Cross-Stage Mapping)
```json
{
  "relation_id": "rel_001",
  "run_id": "run_20260830_mlp",
  "source_op_id": "op_torch_matmul_01",
  "target_op_id": "op_linalg_matmul_04",
  "kind": "LOWERED_TO",
  "confidence": "EXACT",
  "source_type": "COMPILER_METADATA",
  "evidence": ["matching_location", "matching_operands"]
}
```

---

## 7. Capture Strategy

1. **LLVM Pass Capture:**
   Run `opt` with `-print-before-all -print-after-all -print-module-scope` and redirect stderr to a log file.
2. **MLIR Pass Capture:**
   Run `mlir-opt` with `-mlir-print-ir-before-all -mlir-print-ir-after-all` or configure `PassInstrumentation` programmatically.
3. **Trace Collector Script:**
   A Python script (`collector/capture.py`) wraps compilation. It creates a target run directory:
   * Extracts individual snapshots into separate files.
   * Emits a `metadata.json` and a JSONL `events.jsonl` log.

---

## 8. Provenance Strategy

We employ a 5-level provenance detection model to construct the DAG of operations:

1. **Level 1 — Explicit compiler metadata (EXACT / HIGH):**
   Parse source locations like `loc("model.py":12:4)` and MLIR debug attributes. If two ops share the exact location, they map with high confidence.
2. **Level 2 — Structural matching (HIGH):**
   Match signature definitions (operand types, shapes, and attributes) during dialect lowerings (e.g. `linalg.matmul` → `scf.for`).
3. **Level 3 — Dataflow matching (INFERRED):**
   Trace value flow from producers to consumers. If values from `op_a` flow to `op_b` in the before-pass IR, and from `op_a_new` to `op_b_new` in the after-pass IR, we can establish correspondence.
4. **Level 4 — Pattern matching (HEURISTIC):**
   Detect known pattern rewrites (e.g. GEMM + BiasAddition fused into MatMul).
5. **Level 5 — Heuristic similarity (HEURISTIC):**
   Fuzzy score based on name similarity, operand count, and proximity.

---

## 9. UI Plan

A single-page web app built with Vite, React, Tailwind, and React Flow:
* **Timeline (Left):** Vertical pipeline list showing compilation stages and passes.
* **Code Viewer (Bottom):** Two-pane Monaco / CodeMirror editor with syntax highlighting showing before/after diffs of the selected pass.
* **Lineage Graph (Center):** Intersecting dependency graph showing operations lowering and merging down stages.
* **Inspector (Right):** Sidebar showing details, attributes, location, pass timing, and the "Why?" explanation.

---

## 10. Repository Structure

```text
ai-compiler-explorer/
├── PLAN.md                  # This file
├── README.md                # General introduction
├── requirements.txt         # Python dependencies
├── package.json             # Frontend dependencies
│
├── collector/               # Compiler instrumentation & trace capture
│   ├── capture.py           # Main CLI driver
│   ├── llvm_capture.py      # LLVM IR hooks
│   └── mlir_capture.py      # MLIR capture hooks
│
├── parser/                  # Raw IR parser to structured nodes
│   ├── parser_base.py
│   ├── llvm_parser.py
│   └── mlir_parser.py
│
├── provenance/              # Provenance & lineage engine
│   ├── engine.py            # Aggregates L1-L5 matchers
│   └── matchers.py          # Pattern/Dataflow/Metadata matchers
│
├── backend/                 # FastAPI server & SQLite database
│   ├── main.py
│   ├── database.py
│   └── schemas.py
│
├── frontend/                # Vite React app
│   ├── src/
│   │   ├── components/      # DiffViewer, FlowGraph, Inspector, Timeline
│   │   └── App.tsx
│   └── index.html
│
└── examples/                # Compilation models and configs
    ├── test_model.py
    └── run_capture.sh
```

---

## 11. Milestones

* **M0: Environment Setup** (LLVM/MLIR package install, Python 3.12 virtualenv)
* **M1: Minimal Capture Experiment** (Compile test program to multiple stages, save JSONL/snapshots)
* **M2: IR Parser** (Extract operations, dialects, and locations from raw MLIR/LLVM outputs)
* **M3: Provenance Engine** (Calculate EXACT and INFERRED operation mapping across snapshots)
* **M4: Backend API & Storage** (Implement SQLite database and endpoints using FastAPI)
* **M5: Frontend MVP** (Render Stage Timeline, IR Diff Viewer, and Inspector)
* **M6: Lineage Graph** (Draw cross-stage operation lowering paths using React Flow)
* **M7: "Why?" Explanation View** (Generate readable compiler transformation explanations)
* **M8: Scaling and Validation** (Compile & explore larger neural net layers, e.g., Conv2D + Bias + ReLU)

---

## 12. TODO List

- [ ] **M0: Setup**
  1. Install system LLVM and MLIR packages.
  2. Setup Python 3.12 virtual environment.
  3. Verify `opt`, `clang`, `mlir-opt` are in the PATH.
- [ ] **M1: Minimal Capture**
  4. Write a simple C program to compile and trace LLVM passes.
  5. Parse `opt` output logs to separate files per pass.
  6. Create initial run artifact JSONL schema.
- [ ] **M2: Parser**
  7. Implement MLIR text parser to identify block headers, operations, operands, and location attributes.
  8. Implement LLVM parser.
- [ ] **M3: Provenance**
  9. Write Level 1 location matcher.
  10. Write Level 3 dataflow value tracing.
- [ ] **M4: Backend**
  11. Set up FastAPI project.
  12. Implement DB schemas for runs, passes, ops, and relations.
- [ ] **M5: Frontend UI**
  13. Setup Vite-React app.
  14. Integrate CodeMirror/Monaco for IR syntax display.
- [ ] **M6: Graph UI**
  15. Integrate React Flow.
  16. Plot nodes and map relations as edges.

---

## 13. Risks

1. **Python 3.14 Compatibility:**
   * *Risk:* Core packages (PyTorch, JAX, torch-mlir) do not support Python 3.14.
   * *Mitigation:* We will isolate Python compilation scripts to a Python 3.12 environment using virtual environments.
2. **Dialect Boundary Loss of Location:**
   * *Risk:* In mlir lowerings, location tracking might be discarded by specific compiler passes.
   * *Mitigation:* The provenance engine will fall back to dataflow (Level 3) and heuristic (Level 5) matching if `loc(...)` attributes are missing.

---

## 14. Unknowns

* **[TO VERIFY]** Can `mlir-opt` and python-mlir packages be fully installed from Arch pacman repositories without compiling MLIR from source?
* **[TO VERIFY]** How rich are the location attributes preserved by `torch-mlir` during lowering to Linalg and LLVM dialects?
* **[TO VERIFY]** Does torch-mlir provide stable support for Python 3.12 on Linux?

---

## 15. Verification Experiments

1. **Experiment 1:** Install MLIR and LLVM using pacman, verify versions.
2. **Experiment 2:** Run `opt -print-after-all -disable-output` on simple LLVM IR and verify output format.
3. **Experiment 3:** Install Python 3.12 venv and check if `torch` and `torch_mlir` can be successfully imported.
4. **Experiment 4:** Trace PyTorch matrix multiplication to MLIR and print intermediate Linalg representations.
