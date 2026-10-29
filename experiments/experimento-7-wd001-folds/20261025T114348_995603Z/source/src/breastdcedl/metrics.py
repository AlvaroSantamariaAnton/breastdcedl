"""Métricas por paciente: media de probabilidades de sus cortes."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, brier_score_loss)
from torch.utils.data import SequentialSampler
from .model import input_to_device


def patient_metrics(ids, labels, probabilities, threshold):
    raw = pd.DataFrame({'patient_id': list(ids), 'label': list(labels), 'probability': list(probabilities)})
    if raw.empty or not np.isfinite(raw[['label', 'probability']].to_numpy()).all():
        raise ValueError('Predicciones vacías o no finitas.')
    if (raw.groupby('patient_id').label.nunique() != 1).any():
        raise ValueError('Etiquetas inconsistentes por paciente.')
    table = raw.groupby('patient_id', sort=True).agg(label=('label', 'first'), probability=('probability', 'mean'))
    y, p = table.label, table.probability
    if set(y) != {0, 1}:
        raise ValueError('La evaluación requiere ambas clases.')
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {'auc': float(roc_auc_score(y, p)), 'accuracy': float((y == pred).mean()),
               'precision': float(precision_score(y, pred, zero_division=0)),
               'recall': float(recall_score(y, pred, zero_division=0)),
               'specificity': float(tn/(tn+fp)), 'f1': float(f1_score(y, pred, zero_division=0)),
               'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
               'brier': float(brier_score_loss(y, p)),
               'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    metrics.update(precision_no_pcr=float(precision_score(y, pred, pos_label=0, zero_division=0)),
                   recall_no_pcr=float(recall_score(y, pred, pos_label=0, zero_division=0)),
                   f1_no_pcr=float(f1_score(y, pred, pos_label=0, zero_division=0)),
                   balanced_accuracy=float((metrics['recall'] + metrics['specificity']) / 2))
    fpr, tpr, _ = roc_curve(y, p)
    return metrics, table, {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}


def evaluate(model, loader, rows, criterion, device, threshold):
    if not isinstance(loader.sampler, SequentialSampler) or len(loader.dataset) != len(rows):
        raise ValueError('Evaluación: loader secuencial alineado con metadatos.')
    modes = [(m, m.training) for m in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    probs, labels, total = [], [], 0.0
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for x, y in loader:
                x, y = input_to_device(x, device), y.to(device).reshape(-1, 1)
                logits = model(x)
                total += criterion(logits, y).item() * len(y)
                probs.extend(logits.sigmoid().flatten().cpu().tolist())
                labels.extend(y.flatten().cpu().tolist())
        if len(probs) != len(rows) or not np.isfinite(total) or not np.array_equal(labels, rows.pCR.to_numpy()):
            raise ValueError('Evaluación incompleta, no finita o desalineada.')
        metrics, table, roc = patient_metrics(rows.patient_id, labels, probs, threshold)
        metrics['loss'] = total / len(rows)
        return metrics, table, roc
    finally:
        for module, mode in modes:
            module.training = mode
