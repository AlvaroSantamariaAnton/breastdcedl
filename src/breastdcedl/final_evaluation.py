"""Estadística final por paciente; independiente de imágenes y entrenamiento."""
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, brier_score_loss, roc_curve)

from .input import classify


def report(labels, probabilities, threshold, replicas=2000, seed=42):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not set(y).issubset({0, 1}):
        raise ValueError('Se requiere una etiqueta binaria y probabilidad por paciente.')
    pred = np.array([classify(v, threshold) for v in p])
    cm = confusion_matrix(y, pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    result = dict(patients=len(y), threshold=threshold, confusion_matrix=cm.tolist(),
                  class_order=['no-pCR', 'pCR'], precision=precision.tolist(), recall=recall.tolist(),
                  f1=f1.tolist(), support=support.tolist(), macro_f1=float(f1.mean()),
                  accuracy=float(np.mean(y == pred)), brier=float(brier_score_loss(y, p)),
                  balanced_accuracy=float(recall.mean()) if len(set(y)) == 2 else None,
                  auc=None, auc_ci95=None, roc=None)
    if len(set(y)) == 2:
        if replicas < 1:
            raise ValueError('Bootstrap requiere al menos una réplica.')
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(y == k) for k in (0, 1)]
        aucs = []
        for _ in range(replicas):
            ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            aucs.append(roc_auc_score(y[ix], p[ix]))
        result['auc'] = float(roc_auc_score(y, p))
        result['auc_ci95'] = np.quantile(aucs, [.025, .975]).tolist()
        fpr, tpr, _ = roc_curve(y, p)
        result['roc'] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
    result['bootstrap'] = dict(replicas=replicas, seed=seed, method='patient stratified percentile 95%; fixed predictor')
    bins = []
    for k in range(10):
        mask = np.minimum((p*10).astype(int), 9) == k
        if mask.any():
            bins.append(dict(lower=k/10, upper=(k+1)/10, n=int(mask.sum()),
                             mean_probability=float(p[mask].mean()), observed=float(y[mask].mean())))
    result['calibration_bins'] = bins
    return result


def evaluate_patients(table, threshold, replicas=2000):
    required = {'patient_id', 'label', 'probability', 'cohort'}
    if not required.issubset(table.columns) or table.empty or table[list(required)].isna().any().any():
        raise ValueError('Tabla de pacientes incompleta.')
    if table.patient_id.duplicated().any():
        raise ValueError('Debe haber una fila por paciente, no por corte.')
    result = {'overall': report(table.label, table.probability, threshold, replicas),
              'always_no_pcr': report(table.label, np.zeros(len(table)), threshold, replicas),
              'cohorts': {}}
    for cohort, group in table.groupby('cohort'):
        result['cohorts'][str(cohort)] = report(group.label, group.probability, threshold, replicas)
    return result
