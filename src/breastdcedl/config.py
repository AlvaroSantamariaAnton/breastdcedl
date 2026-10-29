"""JSON estricto: los errores de escritura no se ignoran."""
import copy
import json
import math
import re
from pathlib import Path


def keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{where}: campos requeridos exactos: {", ".join(expected)}')


def integer(value, low, high, where):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{where}: entero entre {low} y {high}.')


def number(value, low, high, where, closed=True):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value or not (value <= high if closed else value < high):
        raise ValueError(f'{where}: número finito en el rango permitido.')


def validate(config):
    c = copy.deepcopy(config)
    c.setdefault('preprocessing', {'normalization': 'none'})
    c.setdefault('augmentation', {'horizontal_flip_probability': 0.0})
    c.setdefault('scheduler', None)
    c.setdefault('roi_supervision', None)
    final = c.get('mode') == 'final'
    if 'mode' in c and not final:
        raise ValueError('mode opcional: final.')
    keys(c, ['name', 'seed', 'folds', 'device', 'model', 'training', 'evaluation', 'preprocessing', 'augmentation', 'scheduler', 'roi_supervision'] + (['mode'] if final else []) + (['refit_pilot'] if 'refit_pilot' in c else []), 'config')
    if c['scheduler'] is not None:
        keys(c['scheduler'], ['factor', 'patience', 'min_lr'], 'scheduler')
        number(c['scheduler']['factor'], 0, 1, 'scheduler.factor', closed=False)
        if c['scheduler']['factor'] == 0:
            raise ValueError('scheduler.factor debe ser positivo.')
        integer(c['scheduler']['patience'], 1, 100000, 'scheduler.patience')
        number(c['scheduler']['min_lr'], 0, 10, 'scheduler.min_lr')
    keys(c['augmentation'], ['horizontal_flip_probability'], 'augmentation')
    number(c['augmentation']['horizontal_flip_probability'], 0, 1, 'horizontal_flip_probability')
    keys(c['preprocessing'], ['normalization'], 'preprocessing')
    if c['preprocessing']['normalization'] not in ('none', 'standardize'):
        raise ValueError('preprocessing.normalization: none o standardize.')
    if not isinstance(c['name'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', c['name']):
        raise ValueError('name: 1–64 letras ASCII, números, guion o guion bajo.')
    integer(c['seed'], 0, 2**32-1, 'seed')
    if not isinstance(c['folds'], list) or (not c['folds'] and not final):
        raise ValueError('folds: lista no vacía de 0 a 4.')
    for f in c['folds']:
        integer(f, 0, 4, 'fold')
    if len(set(c['folds'])) != len(c['folds']):
        raise ValueError('No se pueden repetir folds.')
    if c['device'] not in ('cpu', 'cuda'):
        raise ValueError('device: cpu o cuda (también para ROCm).')
    m = c['model']
    if m is None:
        raise ValueError('Arquitectura pendiente: define model en configs/config.json antes de validar o entrenar.')
    m.setdefault('clinical', 'none')
    m.setdefault('attention', 'none')
    keys(m, ['blocks', 'activation', 'adaptive_pool_size', 'hidden_features', 'dropout', 'clinical', 'attention'], 'model')
    if m['attention'] not in ('none', 'spatial'):
        raise ValueError('attention: none o spatial.')
    roi = c['roi_supervision']
    if roi is not None:
        keys(roi, ['path', 'weight'], 'roi_supervision')
        if m['attention'] != 'spatial' or m['clinical'] != 'none':
            raise ValueError('ROI requiere atención espacial y modelo solo imagen.')
        if not isinstance(roi['path'], str) or not roi['path']:
            raise ValueError('Ruta ROI requerida.')
        number(roi['weight'], 0, 1, 'roi_supervision.weight')
        if roi['weight'] == 0: raise ValueError('Peso ROI debe ser positivo.')
    if m['clinical'] not in ('none', 'age_menopause'):
        raise ValueError('model.clinical: none o age_menopause.')
    if m['activation'] not in ('relu', 'leaky_relu', 'gelu'):
        raise ValueError('activation: relu, leaky_relu o gelu.')
    if not isinstance(m['blocks'], list) or not m['blocks']:
        raise ValueError('blocks: lista no vacía.')
    size = 256
    for b in m['blocks']:
        keys(b, ['channels', 'convolutions', 'kernel_size', 'normalization', 'pool'], 'block')
        integer(b['channels'], 1, 4096, 'channels')
        integer(b['convolutions'], 1, 20, 'convolutions')
        integer(b['kernel_size'], 1, 15, 'kernel_size')
        if b['kernel_size'] % 2 != 1:
            raise ValueError('kernel_size debe ser impar (padding que conserva tamaño).')
        if b['normalization'] not in ('none', 'batch', 'group'):
            raise ValueError('normalization: none, batch o group (8 grupos).')
        if b['normalization'] == 'group' and b['channels'] % 8:
            raise ValueError('GroupNorm requiere channels divisible entre 8.')
        if b['pool'] not in ('none', 'max', 'avg'):
            raise ValueError('pool: none, max o avg; tamaño/stride 2.')
        if b['pool'] != 'none':
            size //= 2
        if size < 1:
            raise ValueError('Demasiados pooling: dimensión espacial cero.')
    integer(m['adaptive_pool_size'], 1, size, 'adaptive_pool_size')
    if not isinstance(m['hidden_features'], list):
        raise ValueError('hidden_features debe ser una lista.')
    for v in m['hidden_features']:
        integer(v, 1, 65536, 'hidden_features')
    number(m['dropout'], 0, 1, 'dropout', closed=False)
    t = c['training']
    keys(t, ['batch_size', 'max_epochs', 'patience', 'min_epochs', 'loss', 'optimizer', 'learning_rate', 'weight_decay', 'momentum', 'log_every_batches'] + [k for k in ('loss_unit', 'patients_per_batch') if k in t], 'training')
    if t.get('loss_unit', 'cut') not in ('cut', 'patient'):
        raise ValueError('loss_unit: cut o patient.')
    if t.get('loss_unit') == 'patient':
        integer(t.get('patients_per_batch'), 1, 100000, 'patients_per_batch')
        if roi is not None:
            raise ValueError('Pérdida por paciente no admite supervisión ROI.')
    elif 'patients_per_batch' in t:
        raise ValueError('patients_per_batch requiere loss_unit=patient.')
    for k in ('batch_size', 'max_epochs', 'patience', 'min_epochs', 'log_every_batches'):
        integer(t[k], 1, 100000, k)
    if t['loss'] not in ('normal', 'weighted') or t['optimizer'] not in ('adam', 'adamw', 'sgd'):
        raise ValueError('loss: normal/weighted; optimizer: adam/adamw/sgd.')
    number(t['learning_rate'], 0, 10, 'learning_rate')
    if t['learning_rate'] == 0:
        raise ValueError('learning_rate debe ser positiva.')
    if c['scheduler'] is not None and not 0 < c['scheduler']['min_lr'] <= t['learning_rate']:
        raise ValueError('scheduler.min_lr debe ser positiva y no superar learning_rate.')
    number(t['weight_decay'], 0, 100, 'weight_decay')
    number(t['momentum'], 0, 1, 'momentum', closed=False)
    keys(c['evaluation'], ['threshold'], 'evaluation')
    number(c['evaluation']['threshold'], 0, 1, 'threshold')
    if final and (c['folds'] or c['scheduler'] is not None or roi is not None or m['clinical'] != 'none'):
        raise ValueError('Final: folds vacío, sin scheduler de validación, ROI ni clínica.')
    if 'refit_pilot' in c:
        pilot = c['refit_pilot']
        keys(pilot, ['outer_fold', 'inner_fold'], 'refit_pilot')
        for k in pilot:
            integer(pilot[k], 0, 4, k)
        if pilot['outer_fold'] == pilot['inner_fold'] or c['folds'] != [pilot['inner_fold']]:
            raise ValueError('Piloto: folds=[inner_fold], exterior distinto del interior.')
        if final or c['scheduler'] is not None or roi is not None or m['clinical'] != 'none' or m['attention'] != 'none':
            raise ValueError('Piloto: solo imagen sin atención, scheduler, ROI ni mode final.')
    return c


def load(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f'Campo JSON duplicado: {k}')
            result[k] = v
        return result
    return validate(json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique))
