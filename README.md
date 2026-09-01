# AI Compiler Optimization Explorer

AI Compiler Optimization Explorer is an interactive tool designed to visualize compiler lowering pipelines across PyTorch, Torch-MLIR, Linalg, and LLVM representations.

## Prerequisites

- **Python 3.11 or 3.12**: Standard `torch-mlir` compatibility requires Python 3.11 or 3.12 (Python 3.14 will fail to find matching wheel builds).
- **Node.js & npm**: Required for running the frontend interface.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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
