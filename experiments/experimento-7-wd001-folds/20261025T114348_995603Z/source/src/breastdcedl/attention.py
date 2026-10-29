"""Supervisión débil por cajas comprobadas; nunca entrada de la CNN."""
import math
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def read_boxes(path, rows):
    table = pd.read_csv(path)
    required = ['sample_id', 'patient_id', 'x0', 'y0', 'x1', 'y1']
    if not set(required).issubset(table) or table.sample_id.duplicated().any():
        raise ValueError('CSV ROI inválido o sample_id duplicado.')
    selected = table[table.sample_id.isin(rows.sample_id)].copy()
    owners = rows.set_index('sample_id').patient_id
    for row in selected.itertuples():
        if owners.loc[row.sample_id] != row.patient_id:
            raise ValueError('ROI asignada a otra paciente.')
        coords = (row.x0, row.y0, row.x1, row.y1)
        if not all(math.isfinite(v) for v in coords) or not (0 <= row.x0 < row.x1 <= 256 and 0 <= row.y0 < row.y1 <= 256):
            raise ValueError('Caja ROI inválida o fuera del campo.')
    return selected.set_index('sample_id')


class BoxDataset(Dataset):
    def __init__(self, dataset, rows, boxes, probability, generator):
        self.dataset, self.ids, self.boxes = dataset, list(rows.sample_id), boxes
        self.probability, self.generator = probability, generator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        mask = torch.zeros(1, 256, 256)
        valid = self.ids[index] in self.boxes.index
        if valid:
            b = self.boxes.loc[self.ids[index]]
            mask[:, int(b.y0):int(b.y1), int(b.x0):int(b.x1)] = 1
        if torch.rand((), generator=self.generator).item() < self.probability:
            x, mask = x.flip(-1), mask.flip(-1)
        return x, y, mask, valid


def box_loss(logits, masks, valid):
    valid = valid.bool()
    if not valid.any():
        return logits.sum() * 0
    target = F.adaptive_avg_pool2d(masks[valid], logits.shape[-2:])
    return F.binary_cross_entropy_with_logits(logits[valid], target)
