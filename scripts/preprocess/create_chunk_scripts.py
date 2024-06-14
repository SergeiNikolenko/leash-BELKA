import os
from pathlib import Path

template = """
import pandas as pd
import datamol as dm

from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.store.modelstore import ModelStore

import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

data = pd.read_parquet('../train_chunks/train_chunk_{chunk_num}.parquet')
print(f"Chunk {chunk_num} data load")

smiles_column = "molecule_smiles"

def _preprocess(i, row):

    dm.disable_rdkit_log()

    mol = dm.to_mol(row[smiles_column], ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(
        mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=True, stereo=True
    )

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    return row

chunk_size = 100000  # размер подвыборки для обработки
for start in range(0, len(data), chunk_size):
    sub_data = data.iloc[start:start + chunk_size]
    sub_data = dm.parallelized(_preprocess, sub_data.iterrows(), arg_type="args", n_jobs=-1, progress=True, total=len(sub_data))
    sub_data = pd.DataFrame(sub_data)
    sub_data.drop(columns=['molecule_smiles'], inplace=True)
    sub_data.to_parquet(f'../train_preprocessed/train_chunk_{chunk_num}_preprocessed_{{start // chunk_size}}.parquet')
    print(f"Processed chunk {chunk_num}, part {{start // chunk_size}}")
"""

output_dir = Path('../chunk_scripts')
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 11):
    script_content = template.format(chunk_num=i)
    script_path = output_dir / f'process_chunk_{i}.py'
    with open(script_path, 'w') as file:
        file.write(script_content)
