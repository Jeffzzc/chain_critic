#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

input_path = Path("datasets/MATH500/evaluation_dim.jsonl")
output_path = Path("datasets/MATH500/evaluation_dim_clean.jsonl")

count_in = 0
count_out = 0

with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        count_in += 1
        row = json.loads(line)
        dims = row.get("evaluation_dimensions")
        ok = row.get("ok", False)

        if ok and isinstance(dims, list) and len(dims) > 0:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count_out += 1

print(f"input={count_in}, kept={count_out}, removed={count_in - count_out}")
print(f"saved to: {output_path}")