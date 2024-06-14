import pandas as pd
import os
from pathlib import Path


from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from tqdm import tqdm

data = pd.read_parquet('../train.parquet')
print("data load")

output_dir = Path('../train_chunks')
output_dir.mkdir(parents=True, exist_ok=True)

num_chunks = 10
chunk_size = len(data) // num_chunks + (1 if len(data) % num_chunks != 0 else 0)

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(data))
    chunk = data.iloc[start_idx:end_idx]
    chunk.to_parquet(output_dir / f'train_chunk_{i+1}.parquet')
    print(f'Chunk {i+1} saved, size: {len(chunk)}')
