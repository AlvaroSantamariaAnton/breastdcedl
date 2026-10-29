"""Entrada compartida para archivos externos y evaluación; sin consultar etiquetas."""
import io
import warnings

import numpy as np
from PIL import Image
import torch

PHASES = ('PRE', 'EARLY', 'LATE')
MAX_BYTES = 16 * 1024 * 1024  # Por fase; límite operativo inicial.
MAX_PIXELS = 1024 * 1024


def decode_triplet(phases):
    """Mapping PRE/EARLY/LATE -> bytes PNG. Devuelve 1×3×H×W, PNG/255.

    Las casillas identifican las fases, no los nombres de archivo. No podemos
    certificar alineación anatómica ni orden temporal desde los píxeles.
    """
    if set(phases) != set(PHASES):
        raise ValueError('Se requieren exactamente PRE, EARLY y LATE.')
    arrays = []
    for phase in PHASES:
        data = phases[phase]
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
            raise ValueError(f'{phase}: archivo vacío o superior a 16 MiB.')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != 'PNG' or im.mode != 'L' or getattr(im, 'n_frames', 1) != 1:
                        raise ValueError('Se exige PNG gris de 8 bits, sin animación.')
                    w, h = im.size
                    if min(w, h) < 16 or w*h > MAX_PIXELS:
                        raise ValueError('Cada lado debe medir al menos 16 píxeles; máximo 1.048.576 píxeles.')
                    im.verify()
                with Image.open(io.BytesIO(data)) as im:
                    arrays.append(np.asarray(im, dtype=np.float32) / 255.0)
        except Exception as exc:
            raise ValueError(f'{phase}: PNG no válido: {exc}') from exc
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('Las tres fases deben tener las mismas dimensiones.')
    if any(np.array_equal(arrays[i], arrays[j]) for i in range(3) for j in range(i)):
        raise ValueError('Hay fases duplicadas: dos imágenes contienen exactamente los mismos píxeles.')
    return torch.from_numpy(np.stack(arrays)).unsqueeze(0)


def classify(probability, threshold):
    """Clasificar después de agregar; mismo criterio para app y evaluación."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probabilidad inválida.')
    if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Umbral no fijado o inválido.')
    return int(probability >= threshold)
