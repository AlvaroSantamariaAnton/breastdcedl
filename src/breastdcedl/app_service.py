"""Servicio de la futura interfaz: un triplete externo, sin dataset ni etiquetas."""
import hashlib
import json
import time

from .input import decode_triplet, classify


def analyze_triplet(phases, predictor):
    """No persiste imágenes. Devuelve fases y realce float para visualización.

    La interfaz debe mostrar que esta salida corresponde a UN corte, mientras
    las métricas de test se calcularon agregando los cortes de cada paciente.
    """
    start = time.perf_counter()
    images = decode_triplet(phases)
    probability = predictor.predict_patient([images])
    threshold = predictor.manifest['threshold']
    signature = hashlib.sha256(json.dumps(predictor.manifest, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
    return {
        'probability': probability,
        'predicted': classify(probability, threshold),
        'threshold': threshold,
        'model_checksum': signature,
        'device': str(predictor.device),
        'elapsed_seconds': time.perf_counter() - start,
        'phases': images[0].numpy(),
        'enhancement': (images[0, 1] - images[0, 0]).numpy(),
        'unit': 'single_cut',
    }
