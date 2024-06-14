import pandas as pd
import dask.dataframe as dd
import tqdm as tqdm
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

data = pd.read_parquet('../train.parquet')
print("data load")

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


data = dm.parallelized(_preprocess, data.iterrows(), arg_type="args", n_jobs=-1, progress=True, total=len(data))
data = pd.DataFrame(data)


data.drop(columns=['molecule_smiles'], inplace=True)

data.to_parquet('../train_preprocessed.parquet') 