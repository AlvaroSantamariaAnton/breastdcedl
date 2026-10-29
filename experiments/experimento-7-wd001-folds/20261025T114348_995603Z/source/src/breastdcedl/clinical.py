"""Edad/menopausia con ausencia explícita; ajuste por paciente solo en train."""
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ['age_z', 'menopause', 'age_missing', 'menopause_missing']


def read_clinical(root, ids):
    table = pd.read_csv(root / 'metadata/patients.csv', usecols=['pid', 'age', 'menopause'])
    if table.pid.duplicated().any():
        raise ValueError('IDs clínicos duplicados.')
    table = table.set_index('pid').reindex(list(dict.fromkeys(ids)))
    for field in ('age', 'menopause'):
        table[field] = pd.to_numeric(table[field], errors='raise')
        if np.isinf(table[field].to_numpy()).any():
            raise ValueError(f'{field}: valor infinito.')
    if (table.age.dropna() <= 0).any():
        raise ValueError('Edad debe ser positiva.')
    if not table.menopause.dropna().isin([0, 1]).all():
        raise ValueError('Menopausia debe ser 0/1 o ausente; no se recodifica automáticamente.')
    return table


def fit_clinical(table, train_ids):
    ages = table.loc[list(dict.fromkeys(train_ids)), 'age'].dropna()
    if len(ages) < 2 or ages.std(ddof=0) <= 1e-8:
        raise ValueError('Edades train insuficientes o sin variación.')
    return {'features': FEATURES, 'age_mean': float(ages.mean()),
            'age_std': float(ages.std(ddof=0)), 'age_count': len(ages),
            'scope': 'unique training patients with observed age',
            'missing_policy': 'zero value with explicit missing indicator'}


def encode_clinical(table, stats):
    if stats['features'] != FEATURES or not math.isfinite(stats['age_mean']) or not math.isfinite(stats['age_std']) or stats['age_std'] <= 1e-8:
        raise ValueError('Estadísticas clínicas incompatibles.')
    age_missing = table.age.isna()
    menopause_missing = table.menopause.isna()
    values = np.column_stack([
        ((table.age - stats['age_mean']) / stats['age_std']).fillna(0),
        table.menopause.fillna(0), age_missing, menopause_missing]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Entrada clínica no finita.')
    return dict(zip(table.index, values))


class ClinicalDataset(Dataset):
    def __init__(self, dataset, ids, encoded):
        self.dataset = dataset
        self.values = torch.tensor(np.stack([encoded[pid] for pid in ids]), dtype=torch.float32)
        if len(dataset) != len(self.values):
            raise ValueError('Clínica no alineada con cortes.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return (image, self.values[index]), label
