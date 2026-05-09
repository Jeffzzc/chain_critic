import pandas as pd

parquet_path = "/data/dhf/chain_critic/datasets/Feedback-Bench/data/train-00000-of-00001-eddf1add30d20be1.parquet"
jsonl_path = "/data/dhf/chain_critic/datasets/Feedback-Bench/data/train.jsonl"

df = pd.read_parquet(parquet_path)

# 转成 jsonl
df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

print(f"saved to: {jsonl_path}")
print(f"rows: {len(df)}")
print("columns:", df.columns.tolist())