"""Lotes completos y BCE sobre la media de probabilidades de cada paciente."""
import math
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


class PatientDataset(Dataset):
    def __init__(self, dataset, rows):
        if len(dataset) != len(rows):
            raise ValueError('Dataset y pacientes desalineados.')
        self.dataset = dataset
        self.ids = rows.patient_id.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.ids[index]


class PatientBatchSampler(Sampler):
    def __init__(self, rows, patients_per_batch, generator):
        if type(patients_per_batch) is not int or patients_per_batch < 1:
            raise ValueError('patients_per_batch debe ser entero positivo.')
        if rows.empty or rows.patient_id.isna().any() or not rows.pCR.isin([0, 1]).all():
            raise ValueError('Pacientes/etiquetas inválidos.')
        if (rows.groupby('patient_id').pCR.nunique() != 1).any():
            raise ValueError('Etiquetas contradictorias por paciente.')
        self.groups = list(rows.reset_index(drop=True).groupby('patient_id', sort=True).indices.values())
        self.patients_per_batch, self.generator = patients_per_batch, generator

    def __len__(self):
        return math.ceil(len(self.groups) / self.patients_per_batch)

    def __iter__(self):
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for start in range(0, len(order), self.patients_per_batch):
            yield [int(i) for g in order[start:start+self.patients_per_batch] for i in self.groups[g]]


def patient_logits(logits, labels, ids):
    """logit(media(sigmoid(z))) estable; no media de logits ni de features.

    log(sum(sigmoid(z))) - log(sum(sigmoid(-z))); los tamaños se cancelan.
    La agrupación debe recibir todos los cortes de cada paciente.
    """
    z, y = logits.reshape(-1), labels.reshape(-1)
    if len(z) == 0 or len(z) != len(y) or len(z) != len(ids):
        raise ValueError('Logits, etiquetas e IDs incompatibles.')
    if not torch.isfinite(z).all() or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError('Logits/etiquetas inválidos.')
    groups = {}
    for i, pid in enumerate(ids):
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError('ID de paciente inválido.')
        groups.setdefault(pid, []).append(i)
    pooled, targets = [], []
    for indices in groups.values():
        values, targets_group = z[indices], y[indices]
        if not (targets_group == targets_group[0]).all():
            raise ValueError('Etiquetas contradictorias por paciente.')
        pooled.append(torch.logsumexp(F.logsigmoid(values), 0) - torch.logsumexp(F.logsigmoid(-values), 0))
        targets.append(targets_group[0])
    return torch.stack(pooled).reshape(-1, 1), torch.stack(targets).reshape(-1, 1)
