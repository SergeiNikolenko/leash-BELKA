import pandas as pd
import numpy as np
from pathlib import Path
from molfeat.trans.pretrained import PretrainedHFTransformer
from rdkit import RDLogger
import warnings
import torch
import logging

# Настройка логирования
logging.basicConfig(filename='../process.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Игнорирование предупреждений
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

# Путь к чанку (используем первый чанк для оценки)
chunk_file = '../../data_chunks/chunk_0.parquet'

# Загрузка данных
logging.info("Loading data chunk for batch size estimation...")
data = pd.read_parquet(chunk_file)
logging.info("Data chunk loaded.")

smiles_column = "standard_smiles"
smiles = data[smiles_column].tolist()

# Инициализация преобразователя
transformer_name = 'ChemBERTa-77M-MLM'
transformer = PretrainedHFTransformer(kind=transformer_name, notation='smiles', dtype=np.float32, device='cuda:2', preload=True)

def estimate_batch_size(smiles, transformer):
    try:
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = transformer(smiles)
        logging.info(f"All {len(smiles)} SMILES processed successfully at once.")
        return len(smiles)
    except RuntimeError as e:
        logging.warning(f"Failed to process all {len(smiles)} SMILES at once. Error: {str(e)}")
        low = 100
        high = None
        step = low
        last_successful_batch_size = low

        while step >= 100:
            test_batch_size = low + step
            try:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    _ = transformer(smiles[:test_batch_size])
                logging.info(f"Batch size {test_batch_size} is successful.")
                last_successful_batch_size = test_batch_size
                low = test_batch_size
                if high is not None:
                    step = (high - low) // 2
                else:
                    step *= 2
            except RuntimeError as e:
                logging.warning(f"Batch size {test_batch_size} failed.")
                high = test_batch_size
                step = (high - low) // 2

        return int(last_successful_batch_size * 0.7)

batch_size = estimate_batch_size(smiles, transformer)
logging.info(f"Optimal batch size: {batch_size}")

# Сохранение оптимального размера батча
with open("../../optimal_batch_size.txt", "w") as f:
    f.write(str(batch_size))
logging.info("Optimal batch size saved.")
