import json
import re
import os

INPUT_PATH = "data/raw/prompts_paired_hardpp_200.jsonl"
OUT_PATH = "data/raw/prompts_stealthy_large.jsonl"

def clean_injected_prompt(text):
    # Remove the bracketed headers like [For audit completeness]
    # Regex: \[.*?\] (lazy match)
    # But note: The prompt starts with [SOC], we want to KEEP that.
    # The headers to remove are usually later in the text.
    
    # Strategy: 
    # 1. Keep [SOC] at start.
    # 2. Remove other [...] blocks.
    # 3. Remove "Task:" label.
    # 4. Remove "\n"s.
    
    # Split SOC
    if text.startswith("[SOC]"):
        header = "[SOC]"
        rest = text[5:]
    else:
        header = ""
        rest = text

    # Remove [Internal reviewer note] etc.
    # We want to be careful not to remove things that might be part of the injection if they are bracketed?
    # In this dataset, the headers seem to be effectively distinct.
    # Let's simple remove any [...] that is NOT [SOC].
    
    # Actually, looking at the file: 
    # [SOC] ... \n[For audit completeness] ... \nTask: ...
    
    # Remove the specific bracketed lines.
    rest = re.sub(r'\[.*?\]', '', rest) 
    
    # Remove "Task:"
    rest = rest.replace("Task:", "")
    
    # Clean up whitespace (newlines to spaces)
    rest = rest.replace("\n", " ").strip()
    
    # Collapse multiple spaces
    rest = re.sub(r'\s+', ' ', rest)
    
    return f"{header} {rest}"

def clean_benign_prompt(text):
    # Benign: [SOC] ...
    # Just ensure single line and no weird spaces.
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print(f"Reading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    new_rows = []
    for r in rows:
        p = r["prompt"]
        # Handle int (0/1) or string (benign/injected)
        lbl = r["label"]
        is_injected = (lbl == 1) or (lbl == "injected") or (lbl == "1")
        
        if is_injected:
            new_p = clean_injected_prompt(p)
        else:
            new_p = clean_benign_prompt(p)
            
        new_rows.append({
            "id": r["id"],
            "label": r["label"],
            "prompt": new_p
        })
        
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Wrote {len(new_rows)} lines to {OUT_PATH}")
    
    # Preview
    print("\n--- Preview Benign ---")
    print(new_rows[0]["prompt"])
    print("\n--- Preview Injected ---")
    # Find first injected
    inv = next(r for r in new_rows if (r["label"]==1 or r["label"]=="injected"))
    print(inv["prompt"])

if __name__ == "__main__":
    main()
