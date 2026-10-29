"""Evaluación única de test. NO ejecutar hasta autorización explícita de Álvaro."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from breastdcedl.input import decode_triplet, classify, PHASES, MAX_BYTES
from breastdcedl.inference import FoldEnsemble
from breastdcedl.final_evaluation import evaluate_patients


def infer_patients(rows, root, predictor):
    """Usa el mismo lector de bytes que la futura app, un corte cada vez."""
    records = []
    for patient, group in rows.groupby('patient_id', sort=True):
        if group.pCR.nunique() != 1 or group.cohort.nunique() != 1:
            raise ValueError('Etiquetas o cohortes inconsistentes dentro de una paciente.')
        def batches():
            for row in group.itertuples():
                phases = {}
                for phase in PHASES:
                    path = (root / getattr(row, f'path_{phase.lower()}')).resolve()
                    if not path.is_relative_to(root.resolve()):
                        raise ValueError('Ruta fuera del dataset.')
                    with path.open('rb') as f:
                        phases[phase] = f.read(MAX_BYTES + 1)
                yield decode_triplet(phases)
        probability = predictor.predict_patient(batches())
        records.append(dict(patient_id=patient, label=int(group.pCR.iloc[0]),
                            cohort=str(group.cohort.iloc[0]), cuts=len(group), probability=probability,
                            predicted=classify(probability, predictor.manifest['threshold'])))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-test', action='store_true', help='Solo tras autorización; consume la evaluación reservada.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if not args.execute_test:
        parser.error('Test reservado. Este comando requiere autorización y --execute-test.')
    manifest_path = ROOT / 'experiments/modelo_final.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['threshold'] != .35 or manifest['threshold_status'] != 'approved_by_user_2026-10-28':
        raise ValueError('El umbral debe coincidir con el acuerdo previo al test.')
    run = ROOT / Path(manifest['members'][0]['checkpoint']).parents[2]
    out = run / 'final_test'
    # Directorio exclusivo: un fallo tampoco permite repetir silenciosamente.
    out.mkdir(exist_ok=False)
    files = [manifest_path, Path(__file__), ROOT/'metadata/samples.csv', ROOT/'metadata/patients.csv',
             *sorted((ROOT/'src/breastdcedl').glob('*.py'))]
    audit = dict(state='started', test_used=False, device=args.device, python=platform.python_version(),
                 torch=torch.__version__, bootstrap_replicas=2000, bootstrap_seed=42,
                 hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    def save_audit():
        (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    save_audit()
    (out/'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        predictor = FoldEnsemble(manifest_path, ROOT, args.device)
        samples = pd.read_csv(ROOT/'metadata/samples.csv')
        rows = samples[samples.split == 'test'].copy()
        train = samples[samples.split == 'train']
        if set(rows.patient_id) & set(train.patient_id):
            raise ValueError('Solapamiento de pacientes train/test.')
        if len(rows) != 1758 or rows.patient_id.nunique() != 176 or rows.sample_id.duplicated().any():
            raise ValueError('Test incompleto o duplicado: se esperaban 176 pacientes y 1.758 cortes.')
        if rows[['path_pre','path_early','path_late','pCR']].isna().any().any() or not set(rows.pCR).issubset({0,1}):
            raise ValueError('Metadatos test inválidos.')
        metadata = pd.read_csv(ROOT/'metadata/patients.csv')[['pid','dataset']]
        rows = rows.merge(metadata, left_on='patient_id', right_on='pid', validate='many_to_one').rename(columns={'dataset':'cohort'})
        if len(rows) != 1758 or rows.cohort.isna().any():
            raise ValueError('No se ha podido asignar cohorte a todo el test.')
        audit['test_used'] = True
        save_audit()
        table = infer_patients(rows, ROOT, predictor)
        table.to_csv(out/'patient_scores.csv', index=False)
        # Guardar scores permite reconstruir tablas/curvas sin repetir inferencia.
        result = evaluate_patients(table, manifest['threshold'])
        (out/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        for member in manifest['members']:
            if hashlib.sha256((ROOT/member['checkpoint']).read_bytes()).hexdigest() != member['sha256']:
                raise ValueError('Un checkpoint cambió durante la evaluación.')
        audit['state'] = 'completed'
        audit['patients'], audit['cuts'] = len(table), int(table.cuts.sum())
        save_audit()
        print(f'Evaluación guardada en {out}; no reajustar tras ver el resultado.')
    except Exception as exc:
        audit.update(state='failed', error=str(exc))
        save_audit()
        raise


if __name__ == '__main__':
    main()
