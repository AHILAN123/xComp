import os
import sys
import json
import re

LOG_FILE = "trace.log"
RUN_ID = "run_001"
OUTPUT_DIR = f"runs/{RUN_ID}"
STAGES_DIR = f"{OUTPUT_DIR}/stages/02_linalg"
EVENTS_FILE = f"{OUTPUT_DIR}/events.jsonl"

os.makedirs(STAGES_DIR, exist_ok=True)

events = []
current_pass = None
current_ir = []
ir_count = 0

events.append({"type": "run_start", "run_id": RUN_ID})

with open(LOG_FILE, "r") as f:
    lines = f.readlines()

for line in lines:
    match = re.match(r"// -----// IR Dump (Before|After) (.*?) \((.*?)\) .*? //----- //", line.strip())
    if match:
        # Save previous IR if exists
        if current_pass and current_ir:
            ir_text = "".join(current_ir)
            filename = f"{ir_count:02d}_before_{current_pass}.mlir"
            filepath = os.path.join(STAGES_DIR, filename)
            with open(filepath, "w") as out_f:
                out_f.write(ir_text)
            
            events.append({
                "type": "snapshot",
                "kind": "before",
                "pass": current_pass,
                "path": filepath
            })
            ir_count += 1
            current_ir = []
        
        kind = match.group(1).lower()
        full_name = match.group(2)
        pass_name = match.group(3)
        current_pass = pass_name
        
        events.append({
            "type": "pass_start",
            "pass": pass_name,
            "stage": "linalg"
        })
    else:
        if current_pass is not None:
            current_ir.append(line)

# Handle last block
if current_pass and current_ir:
    ir_text = "".join(current_ir)
    filename = f"{ir_count:02d}_before_{current_pass}.mlir"
    filepath = os.path.join(STAGES_DIR, filename)
    with open(filepath, "w") as out_f:
        out_f.write(ir_text)
    
    events.append({
        "type": "snapshot",
        "kind": "before",
        "pass": current_pass,
        "path": filepath
    })
    events.append({
        "type": "pass_end",
        "pass": current_pass
    })

events.append({"type": "run_end", "run_id": RUN_ID})

with open(EVENTS_FILE, "w") as f:
    for event in events:
        f.write(json.dumps(event) + "\n")

print(f"Extraction complete! Generated {len(events)} events in {EVENTS_FILE}")
