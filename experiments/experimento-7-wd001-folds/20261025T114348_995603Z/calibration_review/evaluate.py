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
    train_patients = samples[samples.fold != fold].drop_duplicates('patient_id')
    table['baseline'] = float(train_patients.pCR.mean())
    tables.append(table)
    audit.append({'fold': fold, 'epoch': state['epoch'], 'last_epoch': last['epoch'], 'completed': True, 'sha256': digest, 'patients': len(table)})
    print(f'Fold {fold}: epoch={state["epoch"]}, AUC={metrics["auc"]:.6f}, recall={metrics["recall"]:.6f}', flush=True)
table = pd.concat(tables)
assert not table.index.duplicated().any() and len(table) == samples.patient_id.nunique()
pooled, _, _ = patient_metrics(table.index, table.label, table.probability, .35)
records.append({'scope': 'pooled_oof', 'threshold': .35, **pooled})

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
metadata = uc.cargar_patients(ROOT)[['pid', 'dataset']]
table = table.reset_index().merge(metadata, left_on='patient_id', right_on='pid', validate='one_to_one', how='left')
assert table.dataset.notna().all()
summary, bins = [], []
for scope, group in [('total', table)] + list(table.groupby('dataset')) + [(f'fold_{k}', g) for k,g in table.groupby('fold')]:
    y, p = group.label.to_numpy(), group.probability.to_numpy()
    row = {'scope': scope, 'patients': len(group), 'observed': float(y.mean()), 'predicted': float(p.mean()), 'brier': brier_score_loss(y,p), 'baseline_brier': brier_score_loss(y,group.baseline), 'log_loss': log_loss(y,p), 'baseline_log_loss': log_loss(y,group.baseline), 'min_p':float(p.min()),'max_p':float(p.max())}
    row['brier_skill'] = 1-row['brier']/row['baseline_brier']
    summary.append(row)
    indices = np.minimum((p*10).astype(int),9)
    ece=0
    for k in range(10):
        mask=indices==k;n=int(mask.sum())
        if not n: continue
        obs=float(y[mask].mean());pred=float(p[mask].mean());z=1.959963984540054
        center=(obs+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(obs*(1-obs)/n+z*z/(4*n*n))/(1+z*z/n)
        bins.append({'scope':scope,'bin_lower':k/10,'bin_upper':(k+1)/10,'n':n,'pcr':int(y[mask].sum()),'mean_probability':pred,'observed':obs,'wilson_low':center-half,'wilson_high':center+half})
        ece+=n/len(group)*abs(obs-pred)
    row['ece_10_fixed_bins']=ece
pd.DataFrame(summary).to_csv(out/'calibration_summary.csv',index=False)
pd.DataFrame(bins).to_csv(out/'reliability_bins.csv',index=False)
assert abs(pooled['brier']-0.20581470512172767)<1e-6
(out/'audit.json').write_text(json.dumps({'checkpoints':audit,'test_used':False,'weights_unchanged':True,'calibrator_fitted':False,'baseline':'prevalence of unique training patients of corresponding fold','bin_edges':list(np.linspace(0,1,11)),'interval':'Wilson 95% descriptive binomial intervals; selected CV models treated as fixed','patients':len(table)},indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False),flush=True)
print(pd.DataFrame(bins).query("scope == 'total'").to_string(index=False),flush=True)
