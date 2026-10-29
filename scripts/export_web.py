"""Exportar el predictor cerrado a ONNX y comprobarlo con tensores sintéticos.

No entrena, no lee imágenes reales ni repite test. Requiere onnx y onnxruntime.
Los archivos se preparan junto al experimento de experiments/; este script no publica nada.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from breastdcedl.inference import FoldEnsemble


class WebPredictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.models = nn.ModuleList(predictor.models)
        self.stats = predictor.stats

    def forward(self, images):
        probs = [torch.sigmoid(model((images-s['mean'])/s['std'])).squeeze(1)
                 for model, s in zip(self.models, self.stats)]
        return torch.stack(probs).mean(dim=0)


def main():
    import onnx
    import onnxruntime as ort
    path = ROOT/'experiments/modelo_final.json'
    predictor = FoldEnsemble(path, ROOT, 'cpu')
    wrapper = WebPredictor(predictor).eval()
    run = ROOT/Path(predictor.manifest['members'][0]['checkpoint']).parents[2]
    out = run/'web_export'
    out.mkdir(exist_ok=False)
    artifact = out/'model.onnx'
    torch.onnx.export(wrapper, (torch.zeros(1,3,256,256),), str(artifact),
                      input_names=['images'], output_names=['probability'],
                      dynamic_axes={'images':{0:'batch',2:'height',3:'width'}, 'probability':{0:'batch'}},
                      opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(artifact))
    session = ort.InferenceSession(str(artifact), providers=['CPUExecutionProvider'])
    generator = torch.Generator().manual_seed(42)
    checks = []
    for h,w in [(256,256),(244,244),(31,49),(257,319),(16,16)]:
        x = torch.rand((1,3,h,w),generator=generator)
        expected = predictor.predict_cuts(x).numpy()
        actual = session.run(None, {'images':x.numpy()})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        assert np.array_equal(actual>=.35,expected>=.35)
        # Vectores de referencia para comprobar después el runtime del navegador.
        fixture = f'input_{h}x{w}.bin'
        x.numpy().astype('<f4').tofile(out/fixture)
        checks.append(dict(shape=list(x.shape),input=fixture,expected=expected.tolist(),
                           max_absolute_error=float(np.max(np.abs(actual-expected)))))
    for member in predictor.manifest['members']:
        assert hashlib.sha256((ROOT/member['checkpoint']).read_bytes()).hexdigest()==member['sha256']
    manifest = dict(model='model.onnx',sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    threshold=.35, input='float32 PNG/255 N,3,H,W PRE/EARLY/LATE; normalization embedded',
                    ensemble='five equal probability weights; one probability per cut',
                    source_manifest=predictor.manifest, checks=checks,
                    onnx_version=onnx.__version__,runtime_version=ort.__version__,
                    browser_verified=False, test_repeated=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Exportación y comprobaciones CPU terminadas: {out}. Falta comprobar navegador.')


if __name__=='__main__':
    main()
