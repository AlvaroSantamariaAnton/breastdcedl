"""Preparar recursos locales de la web; descarga runtime fijo, no publica nada."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import requests

ROOT=Path(__file__).resolve().parents[1]
VERSION='1.31.1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-runtime', action='store_true',
                        help='Descargar de nuevo el runtime; por defecto reutiliza el incluido en web/vendor.')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'experiments/modelo_final.json').read_text(encoding='utf-8'))
    run=ROOT/Path(cfg['members'][0]['checkpoint']).parents[2]
    export=run/'web_export'
    manifest=json.loads((export/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((export/'model.onnx').read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Modelo ONNX alterado.')
    web=ROOT/'web';vendor=web/'vendor';models=web/'model';verification=web/'verification'
    for folder in [vendor,models,verification]:folder.mkdir(exist_ok=True)
    names=['ort.wasm.min.mjs','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']
    if args.download_runtime:
        response=requests.get(f'https://registry.npmjs.org/onnxruntime-web/{VERSION}',timeout=30)
        response.raise_for_status();metadata=response.json()
        response=requests.get(metadata['dist']['tarball'],timeout=120);response.raise_for_status()
        data=response.content
        integrity='sha512-'+base64.b64encode(hashlib.sha512(data).digest()).decode()
        if integrity!=metadata['dist']['integrity']:raise ValueError('Integridad npm incorrecta.')
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for name in names:
                with archive.extractfile('package/dist/'+name) as source:
                    (vendor/name).write_bytes(source.read())
        license_response=requests.get(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/LICENSE',timeout=30)
        license_response.raise_for_status();(vendor/'LICENSE').write_bytes(license_response.content)
        (vendor/'package.json').write_text(json.dumps({'name':'onnxruntime-web','version':VERSION,'integrity':integrity},indent=2),encoding='utf-8')
    else:
        for name in names+['LICENSE','package.json']:
            if not (vendor/name).is_file():
                raise FileNotFoundError(f'Falta web/vendor/{name}; usa --download-runtime para prepararlo.')
        if json.loads((vendor/'package.json').read_text(encoding='utf-8'))['version']!=VERSION:
            raise ValueError('Versión del runtime distinta de la esperada.')
    shutil.copyfile(export/'model.onnx',models/'model.onnx')
    (models/'manifest.json').write_text(json.dumps({'sha256':manifest['sha256'],'threshold':cfg['threshold'],
        'source_manifest_sha256':hashlib.sha256((ROOT/'experiments/modelo_final.json').read_bytes()).hexdigest(),
        'runtime_version':VERSION},indent=2),encoding='utf-8')
    for check in manifest['checks']:shutil.copyfile(export/check['input'],verification/check['input'])
    (verification/'checks.json').write_text(json.dumps(manifest['checks'],indent=2),encoding='utf-8')
    (web/'.nojekyll').touch()
    assets=web/'assets'
    assets.mkdir(exist_ok=True)
    for name in ['curvas_aprendizaje_folds.png','auc_epocas_elegidas.png','curvas_test.png','matriz_confusion.png']:
        source=run/'final_test'/name
        if not source.is_file():
            raise FileNotFoundError(f'Falta la figura del proyecto: {source}')
        shutil.copyfile(source,assets/name)
    for label, source_run in [('normal',run),('ponderada',run/'comparison_weighted')]:
        for kind in ['conjuntas','detalle']:
            name=f'curvas_{kind}_{label}.png'
            shutil.copyfile(source_run/name,assets/name)
    gradcam=run/'web_gradcam'
    if (gradcam/'manifest.json').is_file():
        gm=json.loads((gradcam/'manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256((gradcam/'model-gradcam.onnx').read_bytes()).hexdigest()!=gm['sha256']:
            raise ValueError('Modelo Grad-CAM alterado.')
        shutil.copyfile(gradcam/'model-gradcam.onnx',models/'model-gradcam.onnx')
        (models/'gradcam.json').write_text(json.dumps({'sha256':gm['sha256'],
            'url':'https://huggingface.co/AvaroSantamaria/breastdcedl/resolve/main/model-gradcam.onnx'},indent=2),encoding='utf-8')
        for file in gradcam.glob('*.bin'):shutil.copyfile(file,verification/file.name)
        (verification/'gradcam_checks.json').write_text(json.dumps(gm['checks'],indent=2),encoding='utf-8')
    print('Recursos preparados en web/. Servir solo esa carpeta, no la raíz ni el dataset.')


if __name__=='__main__':main()
