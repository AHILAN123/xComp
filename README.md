# AI Compiler Optimization Explorer

AI Compiler Optimization Explorer is an interactive tool designed to visualize compiler lowering pipelines across PyTorch, Torch-MLIR, Linalg, and LLVM representations.

## Prerequisites

- **Python 3.11 or 3.12**: Standard `torch-mlir` compatibility requires Python 3.11 or 3.12 (Python 3.14 will fail to find matching wheel builds).
- **Node.js & npm**: Required for running the frontend interface.

## Setup

> **Platform requirement:** `torch-mlir` only publishes nightly wheels for **Linux x86_64**.
> This project will **not** install on macOS (Intel or Apple Silicon) or on ARM64 Linux
> (including Apple Silicon Macs running an ARM64 Ubuntu VM).
> Recommended: [GitHub Codespaces](https://github.com/features/codespaces), or a native/emulated x86_64 Ubuntu machine.
> Run `uname -m` in your target environment first — it must print `x86_64`, not `arm64`/`aarch64`.

1. Create and activate a Python virtual environment:
```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
```

2. Install Python dependencies:
```bash
   pip install -r requirements.txt
```
   This installs the exact, known-working versions we've locked in `requirements.txt`.

   If this fails with a "no matching distribution" or "could not find a version" error,
   it likely means the pinned nightly builds have been rotated out of PyTorch/torch-mlir's
   servers (these are dated nightlies and get deleted after a few weeks). Regenerate the lock:
```bash
   pip install -r requirements.in
   pip freeze > requirements.txt
```
   Then commit the refreshed `requirements.txt`.

3. Install frontend dependencies:
```bash
   cd frontend && npm install
```

## Running the Pipeline

1. **Trace Generation**:
   ```bash
   .venv/bin/python experiment2.py 2> trace.log
   ```

2. **Parsing**:
   ```bash
   .venv/bin/python parser.py
   ```
   *Outputs structured traces to `runs/run_001/events.jsonl`.*

## Starting the UI

1. **Start Backend Server**:
   ```bash
   .venv/bin/uvicorn backend_api:app --host 0.0.0.0 --port 8001
   ```

2. **Start Frontend Development Server**:
   ```bash
   cd frontend && npm run dev -- --port 3000
   ```

3. **Access the Application**:
   Open [http://localhost:3000](http://localhost:3000) in your web browser.
