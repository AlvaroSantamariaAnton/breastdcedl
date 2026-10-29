"""Piloto de selección interna y reajuste; exterior observado solo al terminar."""
import copy
import csv
import hashlib
import io
import json

import torch
from torch.utils.data import DataLoader
import utils_caso as uc
from .artifacts import atomic, save_json, confusion_report, prepare_plotting
from .config import validate
from .model import ConfigurableCNN
from .preprocessing import StandardizedDataset
from .metrics import evaluate


def pilot_rows(root, config):
    rows = uc.cargar_samples(root)
    rows = rows[rows.split == 'train'].copy()
    if not rows.fold.isin(range(5)).all() or (rows.groupby('patient_id').fold.nunique() != 1).any():
        raise ValueError('Folds no válidos o pacientes compartidas.')
    if (rows.groupby('patient_id').pCR.nunique() != 1).any():
        raise ValueError('Etiquetas contradictorias.')
    outer, inner = (config['refit_pilot'][k] for k in ('outer_fold', 'inner_fold'))
    a = rows[~rows.fold.isin([outer, inner])].copy()
    b = rows[rows.fold == inner].copy()
    c = rows[rows.fold != outer].copy()
    d = rows[rows.fold == outer].copy()
    for subset in (a, b, c, d):
        if subset.empty or set(subset.pCR) != {0, 1}:
            raise ValueError('Partición vacía o sin ambas clases.')
    if set(a.patient_id) & set(b.patient_id) or set(c.patient_id) & set(d.patient_id):
        raise ValueError('Fuga de pacientes entre particiones.')
    return a, b, c, d


def stage_configs(config, epoch=None):
    c = copy.deepcopy(config)
    del c['refit_pilot']
    c['name'] = config['name'][:53] + ('-inner' if epoch is None else '-refit')
    if epoch is not None:
        c.update(mode='final', folds=[])
        c['training'].update(max_epochs=epoch, min_epochs=epoch)
    return validate(c)


def setup_stage(root, folder, config, hashes):
    folder.mkdir(exist_ok=True)
    path = folder / 'config.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != config:
            raise ValueError('Configuración de etapa incompatible.')
    else:
        save_json(path, config)
    for relative in hashes:
        if relative.endswith('.py'):
            dest = folder / 'source' / relative
            if not dest.exists():
                atomic(dest, (root / relative).read_bytes())


def execute_pilot(root, run, config, hashes, stop, log):
    from .training import run_fold, latest_checkpoint
    a, b, c, d = pilot_rows(root, config)
    fold = config['refit_pilot']['inner_fold']
    inner = run / 'inner_selection'
    cfg = stage_configs(config)
    setup_stage(root, inner, cfg, hashes)
    log('Etapa 1: selección interna; el fold exterior no participa.')
    run_fold(root, inner, cfg, fold, hashes, stop, log, rows_override=(a, b))
    last = latest_checkpoint(inner / f'fold_{fold}')
    if last is None or stop['requested'] or (run / 'STOP').exists():
        return
    state = torch.load(last, map_location='cpu', weights_only=False)
    if not state['completed']:
        return
    epoch = state['best_epoch']
    # La duración depende exclusivamente de la validación interna.
    save_json(run / 'selection.json', dict(epoch=epoch, inner_auc=state['best_auc'],
        outer_used=False, test_used=False, threshold=config['evaluation']['threshold']))
    refit = run / 'refit'
    refit_cfg = stage_configs(config, epoch)
    setup_stage(root, refit, refit_cfg, hashes)
    log(f'Etapa 2: reajuste desde cero durante {epoch} épocas; exterior excluido.')
    run_fold(root, refit, refit_cfg, 'final', hashes, stop, log, rows_override=(c, None))
    final = refit / 'final/checkpoints/final.pt'
    if not final.exists() or stop['requested'] or (run / 'STOP').exists():
        return
    paths = {'inner_selected': inner / f'fold_{fold}/checkpoints/best.pt', 'refit': final}
    evaluate_outer(root, run, config, d, paths, log)


def evaluate_outer(root, run, config, rows, paths, log):
    dest = run / 'outer_evaluation'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    shas = {name: digest(path) for name, path in paths.items()}
    # Resultados completados no se recalculan al reanudar.
    if (dest / 'audit.json').exists():
        previous = json.loads((dest / 'audit.json').read_text())
        if previous['checkpoints'] != shas:
            raise ValueError('Pesos distintos de la evaluación exterior guardada.')
        log(f'Evaluación exterior ya completada: {dest}')
        return
    dest.mkdir(exist_ok=True)
    results, curves = [], {}
    device = torch.device(config['device'])
    threshold = config['evaluation']['threshold']
    for name, path in paths.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        partition = json.loads((path.parent.parent / 'partition.json').read_text())
        forbidden = set(partition['train_patients']) | set(partition['val_patients'])
        if forbidden & set(rows.patient_id):
            raise ValueError('Exterior usado en entrenamiento o selección.')
        ds = uc.BreastDCEDataset(rows, raiz=root)
        if state['normalization']['mode'] == 'standardize':
            ds = StandardizedDataset(ds, state['normalization'])
        model = ConfigurableCNN(state['config']['model']).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        loader = DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
        metrics, _, roc = evaluate(model, loader, rows, torch.nn.BCEWithLogitsLoss(), device, threshold)
        if digest(path) != shas[name] or any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
            raise RuntimeError('Pesos modificados durante evaluación.')
        results.append(dict(model=name, epoch=state['epoch'], threshold=threshold, **metrics))
        curves[name] = roc
        log(f'Exterior {name}: AUC={metrics["auc"]:.4f}; VP={metrics["tp"]}, FP={metrics["fp"]}')
        del model, loader
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=results[0])
    writer.writeheader()
    writer.writerows(results)
    atomic(dest / 'metrics.csv', buffer.getvalue().encode())
    save_json(dest / 'roc.json', curves)
    render_outer(dest, results, curves)
    save_json(dest / 'audit.json', dict(checkpoints=shas, patients=rows.patient_id.nunique(),
        outer_fold=config['refit_pilot']['outer_fold'], threshold=threshold,
        test_used=False, weights_unchanged=True, selection_used_outer=False,
        note='Diagnóstico posterior; folds históricamente utilizados. No evaluación independiente definitiva.'))
    log(f'Piloto terminado: {dest}. No se selecciona modelo automáticamente.')


def render_outer(dest, results, curves):
    prepare_plotting()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for row in results:
        name = row['model']
        roc = curves[name]
        ax.plot(roc['fpr'], roc['tpr'], label=f'{name}: AUC {row["auc"]:.3f}')
        confusion_report(dest / f'{name}_confusion.png', row,
                         f'Piloto · fold exterior · {name} · sin test', row['threshold'])
    ax.plot([0, 1], [0, 1], ':', color='gray')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Tasa de falsos positivos',
           ylabel='Sensibilidad', title='Selección interna frente a reajuste · fold exterior')
    ax.legend()
    fig.savefig(dest / 'comparison.png', dpi=150)
    plt.close(fig)
