"""Exportación Grad-CAM analítica de la cabeza GAP/Linear/ReLU/Linear fijada.
Sin entrenamiento: gradientes respecto a activaciones, nunca actualización de pesos.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from breastdcedl.inference import FoldEnsemble

class ExplainPredictor(nn.Module):
    def __init__(self,predictor):
        super().__init__()
        self.models=nn.ModuleList(predictor.models);self.stats=predictor.stats
        for m in self.models:
            assert m.attention is None and not m.clinical
            assert len(m.hidden)==2 and isinstance(m.hidden[0],nn.Linear) and isinstance(m.hidden[1],nn.ReLU)
            assert m.promedio.output_size==1 and not m.training
    def forward(self,images):
        probabilities=[];positive=[];negative=[]
        for m,s in zip(self.models,self.stats):
            a=(images-s['mean'])/s['std']
            for block in m.bloques:a=block(a)
            pooled=m.promedio(a).flatten(1)
            hidden=m.hidden[0](pooled)
            p=torch.sigmoid(m.salida(torch.relu(hidden))).squeeze(1)
            probabilities.append(p)
            # d p / d pooled channels: exact chain rule in eval mode.
            dz=( (hidden>0).to(hidden.dtype)*m.salida.weight ) @ m.hidden[0].weight
            dp=dz*(p*(1-p)).unsqueeze(1)
            area=a.shape[2]*a.shape[3]
            signed=(a*(dp/area).unsqueeze(-1).unsqueeze(-1)).sum(1)
            positive.append(torch.relu(signed));negative.append(torch.relu(-signed))
        return torch.stack(probabilities).mean(0),torch.stack(negative).mean(0),torch.stack(positive).mean(0)

def reference(predictor,x):
    ps=[];pos=[];neg=[]
    for m,s in zip(predictor.models,predictor.stats):
        with torch.no_grad():
            a=(x-s['mean'])/s['std']
            for block in m.bloques:a=block(a)
        a=a.detach().requires_grad_(True)
        p=torch.sigmoid(m.salida(m.dropout(m.hidden(m.promedio(a).flatten(1))))).squeeze(1)
        g=torch.autograd.grad(p.sum(),a)[0].mean((2,3),keepdim=True)
        signed=(a*g).sum(1)
        ps.append(p.detach());pos.append(signed.relu().detach());neg.append((-signed).relu().detach())
    return torch.stack(ps).mean(0),torch.stack(neg).mean(0),torch.stack(pos).mean(0)

def main():
    import onnx,onnxruntime as ort
    predictor=FoldEnsemble(ROOT/'experiments/modelo_final.json',ROOT,'cpu')
    wrapper=ExplainPredictor(predictor).eval()
    run=ROOT/Path(predictor.manifest['members'][0]['checkpoint']).parents[2]
    out=run/'web_gradcam';out.mkdir(exist_ok=True)
    path=out/'model-gradcam.onnx'
    torch.onnx.export(wrapper,(torch.zeros(1,3,256,256),),str(path),input_names=['images'],output_names=['probability','cam_no_pcr','cam_pcr'],dynamic_axes={'images':{0:'batch',2:'height',3:'width'},'probability':{0:'batch'},'cam_no_pcr':{0:'batch',1:'map_height',2:'map_width'},'cam_pcr':{0:'batch',1:'map_height',2:'map_width'}},opset_version=17,dynamo=False,external_data=False)
    onnx.checker.check_model(str(path));session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    rng=torch.Generator().manual_seed(43);checks=[]
    for n,h,w in [(1,256,256),(1,244,244),(1,31,49),(1,257,319),(1,16,16),(2,32,48)]:
        x=torch.rand(n,3,h,w,generator=rng)
        expected=reference(predictor,x);actual=session.run(None,{'images':x.numpy()})
        base=predictor.predict_cuts(x).numpy()
        np.testing.assert_allclose(actual[0],base,rtol=1e-5,atol=1e-6)
        errors=[]
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b.numpy(),rtol=2e-4,atol=1e-7)
            assert np.isfinite(a).all();errors.append(float(np.max(np.abs(a-b.numpy()))))
        assert np.array_equal(actual[0]>=.35,base>=.35)
        name=f'gradcam_{n}_{h}x{w}'
        x.numpy().astype('<f4').tofile(out/(name+'.bin'))
        for label,arr in zip(['probability','cam_no_pcr','cam_pcr'],expected):arr.numpy().astype('<f4').tofile(out/(name+'_'+label+'.bin'))
        checks.append({'shape':list(x.shape),'input':name+'.bin','prefix':name,'map_shape':list(actual[1].shape),'errors':errors})
    for member in predictor.manifest['members']:
        assert hashlib.sha256((ROOT/member['checkpoint']).read_bytes()).hexdigest()==member['sha256']
    manifest={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model':'model-gradcam.onnx','checks':checks,'layer':'output of last convolutional block, after ReLU and MaxPool, before GAP','target':'p or 1-p; mean of per-member raw positive Grad-CAM maps, no per-member normalization','weights_unchanged':True,'test_repeated':False,'browser_verified':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Small manifest is local; ONNX remains exclusively a future remote download.
    (ROOT/'web/model/gradcam.json').write_text(json.dumps({'sha256':manifest['sha256'],'url':'https://huggingface.co/AvaroSantamaria/breastdcedl/resolve/main/model-gradcam.onnx'},indent=2),encoding='utf-8')
    import shutil
    for file in out.glob('*.bin'):shutil.copyfile(file,ROOT/'web/verification'/file.name)
    (ROOT/'web/verification/gradcam_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(path),'sha256':manifest['sha256'],'checks':checks},indent=2))
if __name__=='__main__':main()
