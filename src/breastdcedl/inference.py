"""Predictor de cinco folds. Solo inferencia; no lee datasets ni decide umbrales."""
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from .model import ConfigurableCNN


class FoldEnsemble:
    """Media de probabilidades, con normalización propia de cada checkpoint."""

    def __init__(self, manifest_path, root, device='cpu'):
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if (manifest['aggregation'] != 'mean_probabilities'
                or manifest['patient_aggregation'] != 'mean'
                or manifest['calibration'] is not None):
            raise ValueError('Procedimiento de inferencia no soportado.')
        if [m['fold'] for m in manifest['members']] != list(range(5)):
            raise ValueError('Se requieren exactamente los folds 0 a 4, en orden.')
        self.device = torch.device(device)
        self.models, self.stats = [], []
        self.manifest = manifest
        for member in manifest['members']:
            data = (Path(root) / member['checkpoint']).read_bytes()
            if hashlib.sha256(data).hexdigest() != member['sha256']:
                raise ValueError(f'Checkpoint alterado: fold {member["fold"]}.')
            # Solo checkpoints locales de confianza, fijados por hash en el manifiesto.
            state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            if state['fold'] != member['fold'] or state['epoch'] != member['epoch']:
                raise ValueError('Fold o época incompatibles con el manifiesto.')
            if state['config'] != manifest['training_config']:
                raise ValueError('Configuración de entrenamiento incompatible.')
            stats = state['normalization']
            if (stats['mode'] != 'standardize' or stats['input_scale'] != 'PNG/255'
                    or not math.isfinite(stats['mean'])
                    or not math.isfinite(stats['std']) or stats['std'] <= 1e-8):
                raise ValueError('Normalización inválida.')
            model = ConfigurableCNN(state['config']['model'])
            model.load_state_dict(state['model'], strict=True)
            model.to(self.device).eval().requires_grad_(False)
            self.models.append(model)
            self.stats.append(stats)

    @torch.inference_mode()
    def predict_members(self, images):
        """N×3×H×W float32 en [0,1] (PNG/255), sin estandarizar. Devuelve 5×N."""
        if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32
                or images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3
                or min(images.shape[2:]) < max(m.minimum_side for m in self.models)):
            raise ValueError('Entrada esperada: float32 N×3×H×W, N>0 y H,W>=16.')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('Entrada esperada: PNG/255 finito en [0,1], sin estandarizar.')
        x = images.to(self.device)
        probabilities = []
        for model, stats in zip(self.models, self.stats):
            model.eval()
            normalized = (x - stats['mean']) / stats['std']
            probabilities.append(torch.sigmoid(model(normalized).squeeze(1)).cpu())
        result = torch.stack(probabilities)
        if not torch.isfinite(result).all():
            raise ValueError('El modelo ha producido probabilidades no finitas.')
        return result

    def predict_cuts(self, images):
        """Una probabilidad por corte; igual peso para las cinco redes."""
        return self.predict_members(images).mean(dim=0)

    def predict_patient(self, batches):
        """Batches de cortes de UNA paciente; media por corte, no media por lote.

        Permite distintos tamaños entre batches. La asociación de los cortes con
        una única paciente es responsabilidad del llamador.
        """
        total, count = 0.0, 0
        for images in batches:
            probabilities = self.predict_cuts(images)
            total += probabilities.double().sum().item()
            count += probabilities.numel()
        if count == 0:
            raise ValueError('La paciente debe aportar al menos un corte.')
        return total / count
