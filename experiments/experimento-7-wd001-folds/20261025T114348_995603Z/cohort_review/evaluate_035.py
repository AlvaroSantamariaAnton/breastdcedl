"""Solo validación de best.pt, sin entrenamiento ni test. Ejecutar desde raíz."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import utils_caso as uc
from breastdcedl.model import ConfigurableCNN
from breastdcedl.preprocessing import StandardizedDataset
from breastdcedl.metrics import evaluate, patient_metrics

run = Path(__file__).resolve().parent.parent
out = Path(__file__).resolve().parent
config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
samples = uc.cargar_samples(ROOT)
samples = samples[samples.split == 'train'].copy()
device = torch.device('cuda')
records, tables, audit = [], [], []
for fold in config['folds']:
    folder = run / f'fold_{fold}'
    path = folder / 'checkpoints/best.pt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = torch.load(path, map_location='cpu', weights_only=False)
    last = torch.load(folder / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert last['completed'] and state['epoch'] == last['best_epoch']
    rows = samples[samples.fold == fold].copy()
    part = json.loads((folder / 'partition.json').read_text())
    assert set(rows.patient_id) == set(part['val_patients'])
    assert not set(part['train_patients']) & set(part['val_patients'])
    model = ConfigurableCNN(config['model']).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ds = StandardizedDataset(uc.BreastDCEDataset(rows, raiz=ROOT), state['normalization'])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    metrics, table, _ = evaluate(model, loader, rows, criterion, device, .35)
    history = pd.read_csv(folder / 'metrics.csv')
    expected = history.loc[history.epoch == state['epoch']].iloc[0]
    assert abs(metrics['auc'] - expected.val_auc) < 1e-6
    assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    records.append({'scope': f'fold_{fold}', 'threshold': .35, 'epoch': state['epoch'], **metrics})
    table['fold'] = fold
    tables.append(table)
    audit.append({'fold': fold, 'epoch': state['epoch'], 'last_epoch': last['epoch'], 'completed': True, 'sha256': digest, 'patients': len(table)})
    print(f'Fold {fold}: epoch={state["epoch"]}, AUC={metrics["auc"]:.6f}, recall={metrics["recall"]:.6f}', flush=True)
table = pd.concat(tables)
assert not table.index.duplicated().any() and len(table) == samples.patient_id.nunique()
pooled, _, _ = patient_metrics(table.index, table.label, table.probability, .35)
records.append({'scope': 'pooled_oof', 'threshold': .35, **pooled})
metadata = uc.cargar_patients(ROOT)[['pid', 'dataset']]
assert not metadata.pid.duplicated().any()
table = table.reset_index().merge(metadata, left_on='patient_id', right_on='pid', validate='one_to_one', how='left')
assert table.dataset.notna().all()
cohorts = []
for cohort, group in table.groupby('dataset'):
    m, _, _ = patient_metrics(group.patient_id, group.label, group.probability, .35)
    cohorts.append({'cohort': cohort, 'patients': len(group), 'pcr': int(group.label.sum()), **m})
    print('COHORT', cohort, json.dumps(cohorts[-1]), flush=True)
pd.DataFrame(cohorts).to_csv(out / 'cohorts_035.csv', index=False)
fold_cohorts = []
for (fold, cohort), group in table.groupby(['fold', 'dataset']):
    m, _, _ = patient_metrics(group.patient_id, group.label, group.probability, .35)
    fold_cohorts.append({'fold': fold, 'cohort': cohort, 'patients': len(group), 'pcr': int(group.label.sum()), **m})
pd.DataFrame(fold_cohorts).to_csv(out / 'cohorts_by_fold_035.csv', index=False)
pd.DataFrame(records).to_csv(out / 'metrics_035.csv', index=False)
expected = pd.read_csv(run / 'threshold_comparison/20261026T005412_420724Z/metrics.csv')
expected = expected[(expected.scope == 'OOF') & (expected.threshold == .35)].iloc[0]
assert all(pooled[k] == expected[k] for k in ('tn','fp','fn','tp'))
assert abs(pooled['auc'] - expected.auc) < 1e-6
(out / 'audit_035.json').write_text(json.dumps({'checkpoints': audit, 'threshold': .35, 'test_used': False, 'weights_unchanged': True, 'aggregation': 'mean probabilities per patient', 'patients': len(table)}, indent=2), encoding='utf-8')
print('POOLED', json.dumps(pooled), flush=True)
