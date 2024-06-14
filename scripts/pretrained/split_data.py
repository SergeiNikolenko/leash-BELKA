import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(filename='../process.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Загрузка данных
data = pd.read_parquet('../../train_preprocessed.parquet')
logging.info("Data loaded.")

# Удаление дубликатов
smiles_column = "standard_smiles"
data_unique = data.drop_duplicates(subset=[smiles_column]).copy()
logging.info("Duplicates removed.")

# Разбивка данных на чанки
chunk_size = 100000  # Измените размер чанка по необходимости
output_dir = Path("../../data_chunks")
output_dir.mkdir(parents=True, exist_ok=True)

for i, start in enumerate(tqdm(range(0, len(data_unique), chunk_size), desc="Splitting data into chunks")):
    chunk = data_unique.iloc[start:start + chunk_size]
    chunk.to_parquet(output_dir / f"chunk_{i}.parquet")
    logging.info(f"Chunk {i} saved.")

logging.info("Data split into chunks.")
