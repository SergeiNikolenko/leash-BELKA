import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(filename='../process.log', level=logging.INFO, format='%(asctime)s - %(message)s')

output_dir = Path("../../pretrained_features")
all_files = list(output_dir.glob("*_features.parquet"))

logging.info(f"Combining {len(all_files)} chunk files.")

df_list = []
for file in tqdm(all_files, desc="Combining chunk files"):
    df_list.append(pd.read_parquet(file))

result_df = pd.concat(df_list, ignore_index=True)
result_df.to_parquet(output_dir / "all_features.parquet")
logging.info("Combined all chunks into all_features.parquet")
