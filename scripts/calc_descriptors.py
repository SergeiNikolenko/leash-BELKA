import pandas as pd
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

data_unique = data.drop_duplicates(subset=[smiles_column]).copy()

data_unique["mol"] = data_unique[smiles_column].parallel_apply(dm.to_mol)
mols = data_unique["mol"].tolist()

num_unique_structures = len(data_unique)
print("Количество уникальных структур после удаления дубликатов:", num_unique_structures)

descriptors = dm.descriptors.batch_compute_many_descriptors(mols,n_jobs=-1, progress=True, batch_size='auto')
descriptors.insert(0, smiles_column, data_unique[smiles_column].values)


import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 16))
axs = axs.flatten()

descriptors_list = ["mw", "fsp3", "n_lipinski_hba", "n_lipinski_hbd", "n_rings", 
                    "n_hetero_atoms", "n_heavy_atoms", "n_rotatable_bonds", 
                    "n_radical_electrons", "tpsa", "sas", "n_aliphatic_carbocycles", 
                    "n_aliphatic_heterocyles", "n_aliphatic_rings", 
                    "n_aromatic_carbocycles", "n_aromatic_heterocyles", 
                    "n_aromatic_rings", "n_saturated_carbocycles", 
                    "n_saturated_heterocyles", "n_saturated_rings"]

for i, descriptor in enumerate(descriptors_list):
    sns.histplot(descriptors, x=descriptor, ax=axs[i])

plt.tight_layout()
plt.savefig('../histograms_train.pdf')

descriptors.to_parquet('../train_descriptors.parquet')