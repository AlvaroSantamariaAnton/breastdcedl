"""Estadísticas compartidas por las tres fases, ajustadas solo en train."""
import math
import torch
from torch.utils.data import Dataset


def fit_standardization(dataset, log=print):
    count, mean, m2 = 0, 0.0, 0.0
    log('Calculando media/desviación solo con imágenes de entrenamiento (sin entrenar)…')
    for i in range(len(dataset)):
        x, _ = dataset[i]  # Entrada ya escalada PNG/255.
        x = x.to(dtype=torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError('Intensidades no finitas al ajustar normalización.')
        n = x.numel()
        variance, average = torch.var_mean(x, correction=0)
        delta = average.item() - mean
        total = count + n
        m2 += variance.item() * n + delta * delta * count * n / total
        mean += delta * n / total
        count = total
        if (i + 1) % 500 == 0 or i + 1 == len(dataset):
            log(f'Normalización: {i + 1}/{len(dataset)} cortes leídos')
    std = math.sqrt(m2 / count) if count else 0.0
    if not math.isfinite(std) or std <= 1e-8:
        raise ValueError('No se puede estandarizar: datos vacíos o desviación nula.')
    return {'mode': 'standardize', 'mean': mean, 'std': std, 'pixels': count,
            'scope': 'train only; all pixels including background; shared PRE/EARLY/LATE',
            'input_scale': 'PNG/255'}


class StandardizedDataset(Dataset):
    def __init__(self, dataset, stats):
        self.dataset = dataset
        self.mean, self.std = stats['mean'], stats['std']
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 1e-8:
            raise ValueError('Estadísticas de normalización inválidas.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return (x - self.mean) / self.std, y


class HorizontalFlipDataset(Dataset):
    """Un sorteo por corte; refleja las tres fases juntas sin interpolación."""
    def __init__(self, dataset, probability, generator):
        self.dataset = dataset
        self.probability = probability
        self.generator = generator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if torch.rand((), generator=self.generator).item() < self.probability:
            x = torch.flip(x, dims=(-1,))
        return x, y
