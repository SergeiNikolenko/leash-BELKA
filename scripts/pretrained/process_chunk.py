import sys
import pandas as pd
import numpy as np
from pathlib import Path
from molfeat.trans.pretrained import PretrainedHFTransformer
from rdkit import RDLogger
import warnings
import torch
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(filename='../process.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Игнорирование предупреждений
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

# Получение пути к файлу чанка из аргументов командной строки
chunk_file = sys.argv[1]
output_dir = Path("../../pretrained_features")
output_dir.mkdir(parents=True, exist_ok=True)

# Загрузка данных
logging.info(f"Loading data from {chunk_file}...")
data = pd.read_parquet(chunk_file)
logging.info(f"Data loaded from {chunk_file}.")

smiles_column = "standard_smiles"
smiles = data[smiles_column].tolist()

# Загрузка оптимального размера батча
with open("../../optimal_batch_size.txt", "r") as f:
    batch_size = int(f.read().strip())
logging.info(f"Using batch size: {batch_size}")

# Инициализация преобразователя
transformer_name = 'ChemBERTa-77M-MLM'
transformer = PretrainedHFTransformer(kind=transformer_name, notation='smiles', dtype=np.float32, device='cuda:2', preload=True)

# Обработка чанка
features = []
for i in tqdm(range(0, len(smiles), batch_size), desc=f"Processing in batches of {batch_size}"):
    batch = smiles[i:i+batch_size]
    try:
        features.extend(transformer(batch))
    except RuntimeError as e:
        logging.error(f"Failed to process batch starting at index {i}: {str(e)}")
        break
    torch.cuda.empty_cache()

features = np.array(features)
if features.ndim > 2:
    features = np.squeeze(features, axis=1)

df = pd.DataFrame({
    smiles_column: smiles[:len(features)],
    transformer_name: list(features)
})
output_chunk_file = output_dir / f"{Path(chunk_file).stem}_features.parquet"
df.to_parquet(output_chunk_file)
logging.info(f"Processed and saved {output_chunk_file}")
