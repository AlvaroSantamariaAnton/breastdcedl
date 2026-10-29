"""Un único diagrama Mermaid, derivado del constructor y del JSON vigente."""
import json
from pathlib import Path


def write_diagram(config_path, output):
    raw = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    lines = ['# Arquitectura vigente', '',
             'Generado desde el JSON de trabajo. Regenerar tras editarlo con',
             '`python scripts/draw_architecture.py`. No representa un modelo entrenado.', '']
    if raw.get('model') is None:
        lines += ['**Arquitectura sin definir.** Pendiente del diseño de Álvaro.', '',
                  'No hay capas ni pesos seleccionados. Validar o entrenar requiere completar `model`.', '']
    else:
        import torch
        from .config import load
        from .model import ConfigurableCNN
        config = load(config_path)
        # Solo formas simbólicas, sin imágenes ni actualizaciones de pesos.
        with torch.random.fork_rng(devices=[]), torch.device('meta'):
            model = ConfigurableCNN(config['model']).eval()
            nodes = ['Entrada PRE / EARLY / LATE · 3 × 256 × 256']
            if config['preprocessing']['normalization'] == 'standardize':
                nodes.append('PNG/255 → (x − media train) / desviación train<br/>Estadísticas comunes a las tres fases')
            handles = []
            def record(layer, inputs, result):
                if layer is model.salida and model.clinical:
                    nodes.append(f'Concatenación: {layer.in_features - 4} características imagen + 4 clínicas = {layer.in_features}')
                if layer is model.attention:
                    nodes.append('Atención: Conv 1x1 → sigmoid → multiplicación de mapas por 2a<br/>Solo imagen; ROI únicamente en pérdida de entrenamiento')
                    return
                label = str(layer).replace('"', "'")
                shape = ' × '.join(map(str, result.shape[1:]))
                nodes.append(f'{label}<br/>Salida: {shape}')
            for layer in model.modules():
                if not list(layer.children()):
                    handles.append(layer.register_forward_hook(record))
            with torch.inference_mode():
                x = torch.empty(1, 3, 256, 256)
                model((x, torch.empty(1, 4)) if model.clinical else x)
            for handle in handles:
                handle.remove()
            count = sum(p.numel() for p in model.parameters())
        lines += [f'Parámetros entrenables: **{count:,}**.', '', '```mermaid', 'flowchart TD']
        nodes += ['Sigmoid → probabilidad del corte', 'Media de probabilidades por paciente']
        for i, label in enumerate(nodes):
            lines.append(f'    n{i}["{label}"]')
            if i:
                lines.append(f'    n{i-1} --> n{i}')
        if model.clinical:
            target = next(i for i, label in enumerate(nodes) if label.startswith('Concatenación'))
            lines += ['    clinical["Edad z + menopausia 0/1 + dos indicadores de ausencia<br/>Edad ajustada solo con pacientes train; valores ausentes a cero"]',
                      f'    clinical --> n{target}']
        lines += ['```', '', 'GAP reduce cada mapa a un valor; solo se retiran los ejes 1 × 1 antes de la densa.',
                  'La densa y su activación preceden a dropout. El logit entra directamente en BCEWithLogitsLoss; sigmoid y agregación se aplican al evaluar.', '']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text('\n'.join(lines), encoding='utf-8')
