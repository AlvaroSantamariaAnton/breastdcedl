import {explain,predict} from './inference.js';
const output=document.getElementById('output'),results=[];
async function floats(name){const r=await fetch('./verification/'+name);if(!r.ok)throw Error('Falta '+name);return new Float32Array(await r.arrayBuffer());}
try{
 const r=await fetch('./verification/gradcam_checks.json');if(!r.ok)throw Error('Faltan referencias Grad-CAM.');
 for(const c of await r.json()){
  if(c.shape[0]!==1)continue;
  const data=await floats(c.input),result=await explain(data,c.shape[2],c.shape[3]),base=await predict(data,c.shape[2],c.shape[3]);
  if(Math.abs(result.probability-base.probability)>1e-6+1e-5*Math.abs(base.probability))throw Error('Predicción distinta');
  const errors=[];
  for(const [i,label] of ['cam_no_pcr','cam_pcr'].entries()){
   const expected=await floats(c.prefix+'_'+label+'.bin'),actual=result.maps[i];
   if(actual.length!==expected.length)throw Error('Dimensión distinta');
   let error=0;
   for(let k=0;k<actual.length;k++){const e=Math.abs(actual[k]-expected[k]);if(e>1e-7+2e-4*Math.abs(expected[k]))throw Error('Grad-CAM difiere de autograd');error=Math.max(error,e);}
   errors.push(error);
  }
  results.push({shape:c.shape,map:[result.height,result.width],errors});output.textContent=JSON.stringify({state:'running',results},null,2);
 }
 output.textContent=JSON.stringify({state:'passed',results},null,2);
}catch(e){output.textContent=JSON.stringify({state:'failed',error:e.message,results},null,2);}
