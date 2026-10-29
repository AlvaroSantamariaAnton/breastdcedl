"""Entrenamiento solo por orden de Álvaro; entrada JSON, nunca usa test."""
import hashlib
import math
import platform
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils_caso as uc
from .artifacts import atomic, save_checkpoint, save_history, save_json, plot_epoch, plot_summary
from .metrics import evaluate
from .model import ConfigurableCNN, input_to_device
from .preprocessing import fit_standardization, StandardizedDataset, HorizontalFlipDataset
from .patient_loss import PatientDataset, PatientBatchSampler, patient_logits


def fingerprint(root):
    paths = list((root / 'src/breastdcedl').glob('*.py'))
    paths += [root / 'utils_caso.py', root / 'scripts/train.py', root / 'metadata/samples.csv', root / 'metadata/patients.csv']
    return {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment(device):
    return {'host': platform.node(), 'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'device': str(device), 'cuda': torch.version.cuda, 'hip': torch.version.hip,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}


def split_rows(root, fold):
    # Las etiquetas test no participan en selección, métricas ni inferencia.
    samples = uc.cargar_samples(root)
    samples = samples[samples.split == 'train']
    if (samples.groupby('patient_id').fold.nunique() != 1).any():
        raise ValueError('Una paciente aparece en varios folds.')
    if (samples.groupby('patient_id').pCR.nunique() != 1).any():
        raise ValueError('Etiquetas contradictorias por paciente.')
    train, val = uc.particion(samples, fold)
    if set(train.pCR) != {0, 1} or set(val.pCR) != {0, 1}:
        raise ValueError('Cada partición debe contener ambas clases.')
    return train, val


def final_rows(root):
    rows = uc.cargar_samples(root)
    rows = rows[rows.split == 'train'].copy()
    if rows.empty or set(rows.pCR) != {0, 1} or (rows.groupby('patient_id').pCR.nunique() != 1).any():
        raise ValueError('Desarrollo vacío o etiquetas inconsistentes.')
    if not rows.fold.isin(range(5)).all():
        raise ValueError('Folds de desarrollo inválidos.')
    return rows


def make_optimizer(model, config):
    kwargs = {'lr': config['learning_rate'], 'weight_decay': config['weight_decay']}
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), momentum=config['momentum'], **kwargs)
    cls = torch.optim.Adam if config['optimizer'] == 'adam' else torch.optim.AdamW
    return cls(model.parameters(), **kwargs)


def make_scheduler(optimizer, config):
    if config is None:
        return None
    # PyTorch reduce cuando num_bad_epochs > patience. JSON cuenta épocas exactas.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config['factor'], patience=config['patience'] - 1,
        threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=config['min_lr'], eps=0.0)


def train_epoch(model, loader, optimizer, criterion, device, epoch, every, log):
    model.train()
    total, count = 0.0, 0
    start = time.perf_counter()
    for index, batch in enumerate(loader, 1):
        x, y = batch[:2]
        x, y = input_to_device(x, device), y.to(device).reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        if len(batch) == 4:
            from .attention import box_loss
            logits, maps = model(x, return_attention=True)
            aux = box_loss(maps, batch[2].to(device), batch[3].to(device))
            primary = criterion(logits, y)
            loss = primary + model.roi_weight * aux
            if index % every == 0 or index == len(loader):
                log(f'BCE pCR={primary.item():.5f} | ROI={aux.item():.5f} | peso={model.roi_weight:g}')
        elif len(batch) == 3:
            logits, y = patient_logits(model(x), y, batch[2])
            loss = criterion(logits, y)
        else:
            loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Pérdida no finita; esta época no se guarda.')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Gradientes no finitos; esta época no se guarda.')
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
        if index % every == 0 or index == len(loader):
            log(f'Época {epoch} | lote {index}/{len(loader)} | loss optimización={total/count:.5f} | {time.perf_counter()-start:.1f}s')
    return total/count


def capture_rng(generators, device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'loaders': [g.get_state() for g in generators],
            'gpu': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, generators, device, previous_env, current_env, seed, log):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    for generator, rng in zip(generators, state['loaders']):
        generator.set_state(rng)
    if device.type == 'cuda':
        same = all(previous_env[k] == current_env[k] for k in ('gpu', 'torch', 'cuda', 'hip'))
        if same and state['gpu'] is not None:
            torch.cuda.set_rng_state(state['gpu'], device)
        else:
            torch.cuda.manual_seed_all(seed)
            log('Cambio de plataforma: se reinicia RNG GPU; no se garantiza identidad numérica.')


def latest_checkpoint(folder):
    # Prioriza numerados; tolera corte después de guardar epoch_N antes de last.pt.
    paths = sorted((folder / 'checkpoints').glob('epoch_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    return paths[-1] if paths else None


def run_fold(root, run, config, fold, hashes, stop, log, rows_override=None):
    t = config['training']
    final = config.get('mode') == 'final'
    roi = config.get('roi_supervision')
    hashes = dict(hashes)
    if roi:
        hashes['roi_csv'] = hashlib.sha256((root / roi['path']).read_bytes()).hexdigest()
    device = torch.device(config['device'])
    folder = run / ('final' if final else f'fold_{fold}')
    folder.mkdir(exist_ok=True)
    checkpoint_path = latest_checkpoint(folder)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path else None
    if state and (state['config'] != config or state['sources'] != hashes or state['fold'] != fold):
        raise ValueError('Checkpoint incompatible con configuración/fuentes/fold; no se sobrescribe.')
    if state:
        atomic(folder / 'checkpoints/last.pt', checkpoint_path.read_bytes())
        if not final:
            best_path = folder / 'checkpoints' / f'epoch_{state["best_epoch"]:03d}.pt'
            atomic(folder / 'checkpoints/best.pt', best_path.read_bytes())
    if state and state['completed']:
        publish(folder, state, log)
        log(f'Fold {fold} ya terminado, se conserva.')
        return
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, val = rows_override if rows_override is not None else ((final_rows(root), None) if final else split_rows(root, fold))
    generators = [torch.Generator().manual_seed(config['seed']) for _ in range(3)]
    datasets = [uc.BreastDCEDataset(rows, raiz=root) for rows in ((train, train) if final else (train, train, val))]
    mode = config.get('preprocessing', {}).get('normalization', 'none')
    stats = state['normalization'] if state else (
        fit_standardization(datasets[0], log) if mode == 'standardize' else {'mode': 'none'})
    if stats['mode'] != mode:
        raise ValueError('Normalización del checkpoint incompatible.')
    if mode == 'standardize':
        datasets = [StandardizedDataset(ds, stats) for ds in datasets]
        log(f'Normalización común: media={stats["mean"]:.8f}, desviación={stats["std"]:.8f}')
    save_json(folder / 'normalization.json', stats)
    probability = config.get('augmentation', {}).get('horizontal_flip_probability', 0.0)
    if probability or roi:
        # RNG independiente, incluido en capture_rng/restore_rng junto a los loaders.
        generators.append(torch.Generator().manual_seed(config['seed']))
        if roi:
            from .attention import BoxDataset, read_boxes
            boxes = read_boxes(root / roi['path'], train)
            if boxes.empty: raise ValueError('No hay cajas comprobadas en este fold de entrenamiento.')
            save_json(folder / 'roi_coverage.json', {'samples': len(boxes), 'patients': int(boxes.patient_id.nunique()), 'weight': roi['weight']})
            atomic(folder / 'roi_train.csv', boxes.reset_index().to_csv(index=False).encode('utf-8'))
            datasets[0] = BoxDataset(datasets[0], train, boxes, probability, generators[-1])
            log(f'ROI solo train: {len(boxes)}/{len(train)} cortes; resto conserva BCE.')
        else:
            datasets[0] = HorizontalFlipDataset(datasets[0], probability, generators[-1])
        log(f'Aumento solo en entrenamiento: volteo horizontal conjunto, p={probability:g}. Evaluación sin volteos.')
    clinical_stats = None
    if config['model'].get('clinical', 'none') == 'age_menopause':
        from .clinical import read_clinical, fit_clinical, encode_clinical, ClinicalDataset
        table = read_clinical(root, list(train.patient_id) + list(val.patient_id))
        clinical_stats = state['clinical_stats'] if state else fit_clinical(table, train.patient_id)
        encoded = encode_clinical(table, clinical_stats)
        datasets = [ClinicalDataset(ds, rows.patient_id, encoded) for ds, rows in zip(datasets, (train, train, val))]
        save_json(folder / 'clinical.json', clinical_stats)
        log('Clínica: edad/menopausia e indicadores de ausencia; todas las pacientes conservadas.')
    loaders = [DataLoader(ds, batch_size=t['batch_size'], shuffle=i == 0,
                          num_workers=0, generator=generators[i]) for i, ds in enumerate(datasets)]
    loss_unit = t.get('loss_unit', 'cut')
    if loss_unit == 'patient':
        loaders[0] = DataLoader(PatientDataset(datasets[0], train),
            batch_sampler=PatientBatchSampler(train, t['patients_per_batch'], generators[0]),
            num_workers=0, generator=generators[0])
        log(f'BCE por paciente; {t["patients_per_batch"]} pacientes completas por lote; {len(loaders[0])} pasos/época. Evaluación: lotes de {t["batch_size"]} cortes.')
    model = ConfigurableCNN(config['model']).to(device)
    model.roi_weight = roi['weight'] if roi else 0.0
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, config.get('scheduler'))
    weight = uc.pos_weight(train.drop_duplicates('patient_id') if loss_unit == 'patient' else train) if t['loss'] == 'weighted' else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=None if weight is None else torch.tensor(weight, device=device))
    env = environment(device)
    history, best_auc, best_epoch, wait, first = [], -math.inf, None, 0, 1
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        history = state['history']
        best_auc, best_epoch, wait, first = state['best_auc'], state['best_epoch'], state['wait'], state['epoch'] + 1
        restore_rng(state['rng'], generators, device, state['environment'], env, config['seed']+first, log)
        publish(folder, state, log)
    else:
        save_json(folder / 'partition.json', {'train_patients': sorted(train.patient_id.unique().tolist()),
                  'val_patients': [] if final else sorted(val.patient_id.unique().tolist()), 'pos_weight': weight,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'input': 'PRE/EARLY/LATE, float32, PNG/255, 3x256x256',
                  'normalization': stats, 'aggregation': 'mean probabilities'})
        atomic(folder / 'model.txt', str(model).encode('utf-8'))
    batch_description = f'{t["patients_per_batch"]} pacientes' if loss_unit == 'patient' else f'{t["batch_size"]} cortes'
    log(f'Fold {fold}: train={train.patient_id.nunique()} pacientes, validación={0 if final else val.patient_id.nunique()}, lote train={batch_description}, inicio={first}')
    for epoch in range(first, t['max_epochs']+1):
        if stop['requested'] or (run / 'STOP').exists() or (stop.get('file') is not None and stop['file'].exists()):
            break
        start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr_used = optimizer.param_groups[0]['lr']
        optimization_loss = train_epoch(model, loaders[0], optimizer, criterion, device, epoch, t['log_every_batches'], log)
        train_seconds = time.perf_counter()-start
        log(f'Fold {fold} época {epoch}: evaluando {"solo train" if final else "train y validación"} por paciente, sin actualizar pesos…')
        train_metrics, _, train_roc = evaluate(model, loaders[1], train, criterion, device, config['evaluation']['threshold'], loss_unit=loss_unit)
        if final:
            history.append({'epoch': epoch, 'optimization_loss': optimization_loss, 'learning_rate': lr_used,
                            **{f'train_{k}': v for k, v in train_metrics.items()}})
            state = {'format_version': 1, 'config': config, 'sources': hashes, 'environment': env,
                     'fold': fold, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'rng': capture_rng(generators, device), 'history': history, 'best_auc': None,
                     'best_epoch': None, 'wait': 0, 'completed': epoch == t['max_epochs'],
                     'scheduler': None, 'normalization': stats, 'clinical_stats': None, 'roc': {'train': train_roc}}
            save_checkpoint(folder, state, False)
            publish(folder, state, log)
            log(f'FINAL época {epoch}/{t["max_epochs"]} GUARDADA | loss={train_metrics["loss"]:.5f} | AUC train={train_metrics["auc"]:.4f} | sin validación/test')
            if state['completed'] or stop['requested']: break
            continue
        val_metrics, _, val_roc = evaluate(model, loaders[2], val, criterion, device, config['evaluation']['threshold'], loss_unit=loss_unit)
        improved = val_metrics['auc'] > best_auc
        if improved:
            best_auc, best_epoch, wait = val_metrics['auc'], epoch, 0
        else:
            wait += 1
        completed = epoch == t['max_epochs'] or (epoch >= t['min_epochs'] and wait >= t['patience'])
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
        lr_next = optimizer.param_groups[0]['lr']
        if lr_next < lr_used:
            log(f'Learning rate reducido: {lr_used:g} → {lr_next:g}, efectivo desde la siguiente época.')
        row = {'epoch': epoch, 'optimization_loss': optimization_loss, 'learning_rate': lr_used, 'next_learning_rate': lr_next,
               **{f'train_{k}': v for k, v in train_metrics.items()},
               **{f'val_{k}': v for k, v in val_metrics.items()},
               'train_seconds': train_seconds, 'seconds_before_save': time.perf_counter()-start,
               'memory_max_mib': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0.0}
        history.append(row)
        state = {'format_version': 1, 'config': config, 'sources': hashes, 'environment': env,
                 'fold': fold, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'rng': capture_rng(generators, device), 'history': history, 'best_auc': best_auc,
                 'best_epoch': best_epoch, 'wait': wait, 'completed': completed,
                 'scheduler': scheduler.state_dict() if scheduler is not None else None,
                 'normalization': stats, 'clinical_stats': clinical_stats, 'roc': {'train': train_roc, 'val': val_roc}}
        save_checkpoint(folder, state, improved)
        publish(folder, state, log)
        log(f'Fold {fold} | época {epoch} GUARDADA | loss={train_metrics["loss"]:.5f} val_loss={val_metrics["loss"]:.5f} '
            f'| AUC={train_metrics["auc"]:.4f}/{val_metrics["auc"]:.4f} '
            f'| precisión={train_metrics["precision"]:.4f}/{val_metrics["precision"]:.4f} '
            f'| accuracy={train_metrics["accuracy"]:.4f}/{val_metrics["accuracy"]:.4f} '
            f'| lr={lr_used:g} siguiente_lr={lr_next:g} | mejor={best_auc:.4f} en {best_epoch} | paciencia={wait}/{t["patience"]} | total={time.perf_counter()-start:.1f}s')
        if completed or stop['requested'] or (run / 'STOP').exists() or (stop.get('file') is not None and stop['file'].exists()):
            break


def publish(folder, state, log):
    # El dibujo y las importaciones no deben alterar el siguiente entrenamiento.
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        _publish(folder, state, log)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)


def _publish(folder, state, log):
    """Salidas derivadas recuperables desde un checkpoint ya guardado."""
    save_history(folder, state['history'])
    epoch = state['epoch']
    save_json(folder / 'roc' / f'epoch_{epoch:03d}.json', state['roc'])
    if state['config'].get('mode') == 'final':
        from .final_plots import plot_final
        if state['completed']:
            atomic(folder / 'checkpoints/final.pt', (folder / 'checkpoints' / f'epoch_{epoch:03d}.pt').read_bytes())
        try:
            plot_final(folder, state)
        except Exception as exc:
            log(f'AVISO: checkpoint guardado, figuras pendientes: {exc}')
        return
    try:
        plot_epoch(folder, state['history'], state['roc'], state['best_epoch'])
        plot_summary(folder.parent)
    except Exception as exc:
        log(f'AVISO: checkpoint guardado, dibujo pendiente: {exc}. Regenerar con scripts/plot_run.py.')


def execute(root, run, config, hashes, log):
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU no disponible; no se cambia a CPU automáticamente.')
    stop = {'requested': False, 'file': run / 'STOP'}
    def request_stop(signum, frame):
        stop['requested'] = True
        log('Parada solicitada: espera a que termine y se guarde la época actual.')
    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        if (run / 'STOP').exists():
            raise ValueError('Retira el archivo STOP para reanudar.')
        if config.get('refit_pilot'):
            from .refit_pilot import execute_pilot
            execute_pilot(root, run, config, hashes, stop, log)
            return
        for fold in (['final'] if config.get('mode') == 'final' else config['folds']):
            if stop['requested'] or (run / 'STOP').exists():
                break
            run_fold(root, run, config, fold, hashes, stop, log)
    finally:
        signal.signal(signal.SIGINT, previous)
    log(f'Secuencia finalizada o pausada. Resultados: {run}')
