import pandas as pd
import numpy as np
from pathlib import Path
from molfeat.trans.pretrained import PretrainedHFTransformer
from rdkit import RDLogger
import warnings
import torch
from tqdm import tqdm

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
# Игнорирование предупреждений
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_parquet('../train_preprocessed.parquet')
print("Data loaded")
smiles_column = "standard_smiles"
data_unique = data.drop_duplicates(subset=[smiles_column]).copy()
smiles = data_unique[smiles_column].tolist()

# Папка для сохранения результатов
output_dir = Path("../pretrained_features")
output_dir.mkdir(parents=True, exist_ok=True)

# Инициализация преобразователя
transformer_name = 'ChemBERTa-77M-MLM'
transformer = PretrainedHFTransformer(kind=transformer_name, notation='smiles', dtype=np.float32, device='cuda:2', preload=True)

# Оценка размера батча
def estimate_batch_size():
    try:
        # Try to process all data at once
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = transformer(smiles)
        print(f"All {len(smiles)} SMILES processed successfully at once.")
        return len(smiles)
    except RuntimeError as e:
        print(f"Failed to process all {len(smiles)} SMILES at once. Error: {str(e)}")
        # If fail, proceed with incremental batch size estimation
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
                print(f"Batch size {test_batch_size} is successful.")
                last_successful_batch_size = test_batch_size
                low = test_batch_size
                if high is not None:
                    step = (high - low) // 2
                else:
                    step *= 2
            except RuntimeError as e:
                print(f"Batch size {test_batch_size} failed")
                high = test_batch_size
                step = (high - low) // 2

        return int(last_successful_batch_size * 0.7)

batch_size = estimate_batch_size()
print(f"Optimal batch size: {batch_size}")



batch_size = estimate_batch_size()
print(f"Optimal batch size: {batch_size}")

features = []
for i in tqdm(range(0, len(smiles), batch_size), desc=f"Processing in batches of {batch_size}"):
    batch = smiles[i:i+batch_size]
    try:
        features.extend(transformer(batch))
    except RuntimeError as e:
        print(f"Failed to process batch starting at index {i}: {str(e)}")
        break
    torch.cuda.empty_cache()

features = np.array(features)
if features.ndim > 2:
    features = np.squeeze(features, axis=1)

df = pd.DataFrame({
    smiles_column: smiles[:len(features)],
    transformer_name: list(features)
})
df.to_parquet(output_dir / f"{transformer_name}_features.parquet")
print("Done")