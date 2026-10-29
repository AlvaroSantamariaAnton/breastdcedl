"""Entrada principal: python scripts/train.py --config configs/config.json."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path, help='JSON; inicia una ejecución nueva.')
    group.add_argument('--resume', type=Path, help='Carpeta de una ejecución de este nuevo formato.')
    parser.add_argument('--validate', action='store_true', help='Validar config/modelo/particiones sin entrenar ni crear un run.')
    args = parser.parse_args()
    from breastdcedl.config import load
    from breastdcedl.model import ConfigurableCNN
    from breastdcedl.training import execute, fingerprint, split_rows
    from breastdcedl.artifacts import atomic, save_json
    try:
        config = load(args.config if args.config else args.resume / 'config.json')
    except ValueError as error:
        parser.error(str(error))
    hashes = fingerprint(ROOT)
    model = ConfigurableCNN(config['model'])
    print(model)
    print(f'Parámetros: {sum(p.numel() for p in model.parameters()):,}')
    del model
    if config.get('mode') == 'final':
        from breastdcedl.training import final_rows
        rows = final_rows(ROOT)
        print(f'FINAL: {rows.patient_id.nunique()} pacientes / {len(rows)} cortes; {config["training"]["max_epochs"]} épocas fijas. Sin validación ni test.')
    if config.get('refit_pilot'):
        from breastdcedl.refit_pilot import pilot_rows
        inner_train, inner_val, refit_train, outer = pilot_rows(ROOT, config)
        print(f'PILOTO: selección {inner_train.patient_id.nunique()} train / {inner_val.patient_id.nunique()} validación interna; reajuste {refit_train.patient_id.nunique()} train; exterior {outer.patient_id.nunique()} pacientes. Test excluido.')
    for fold in ([] if config.get('refit_pilot') else config['folds']):
        train, val = split_rows(ROOT, fold)
        print(f'Fold {fold}: {train.patient_id.nunique()} pacientes train / {val.patient_id.nunique()} validación')
        if config['training'].get('loss_unit') == 'patient':
            import torch
            from breastdcedl.patient_loss import PatientBatchSampler
            sampler = PatientBatchSampler(train, config['training']['patients_per_batch'], torch.Generator().manual_seed(config['seed']))
            print(f'BCE paciente: {sampler.patients_per_batch} pacientes completas/lote; {len(sampler)} pasos/época; evaluación {config["training"]["batch_size"]} cortes/lote.')
        if config.get('roi_supervision'):
            from breastdcedl.attention import read_boxes
            boxes = read_boxes(ROOT / config['roi_supervision']['path'], train)
            if boxes.empty: raise ValueError('No hay ROI verificadas para train en este fold.')
            print(f'ROI verificadas solo train: {len(boxes)} cortes / {boxes.patient_id.nunique()} pacientes')
        if config['model'].get('clinical') == 'age_menopause':
            from breastdcedl.clinical import read_clinical, fit_clinical, encode_clinical
            table = read_clinical(ROOT, list(train.patient_id) + list(val.patient_id))
            stats = fit_clinical(table, train.patient_id)
            encode_clinical(table, stats)
            print(f'Clínica validada: edad ajustada con {stats["age_count"]} pacientes train; faltantes conservados con indicadores.')
    if args.resume:
        run = args.resume.resolve()
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['sources'] != hashes or manifest['config'] != config:
            raise ValueError('Fuentes/metadatos o configuración cambiaron. Conserva el run y crea otro; no se reanuda con cambios.')
    if args.validate:
        print('Validación correcta. No se ha entrenado, actualizado pesos ni creado una ejecución.')
        return
    if args.config:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run = ROOT / 'runs' / config['name'] / stamp
        run.mkdir(parents=True, exist_ok=False)
        save_json(run / 'config.json', config)
        save_json(run / 'manifest.json', {'format_version': 1, 'sources': hashes, 'config': config,
                  'normalization': config['preprocessing'], 'aggregation': 'mean probabilities per patient'})
        for relative in hashes:
            if relative.endswith('.py'):
                atomic(run / 'source' / relative, (ROOT / relative).read_bytes())
    def log(message):
        print(message, flush=True)
    log(f'Run: {run}\nConfiguración congelada: {run / "config.json"}')
    execute(ROOT, run, config, hashes, log)


if __name__ == '__main__':
    main()
