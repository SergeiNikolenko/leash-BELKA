import pandas as pd
from pathlib import Path
from tqdm import tqdm

input_dir = Path('../train_preprocessed')
files = list(input_dir.glob('train_chunk_*.parquet'))
files.sort()

combined_df_list = []

for file in tqdm(files, desc="Processing files"):
    df = pd.read_parquet(file)
    combined_df_list.append(df)

combined_df = pd.concat(combined_df_list, ignore_index=True)
output_file = '../train_combined_preprocessed.parquet'
combined_df.to_parquet(output_file)

print(f"Combined file saved as {output_file}")
