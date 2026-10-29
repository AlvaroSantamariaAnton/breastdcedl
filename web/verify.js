import {predict,sha256} from './inference.js';
import {prepareTriplet,decodePNG} from './png.js';
const output=document.getElementById('output'),results=[];
async function bytes(name){const r=await fetch('./verification/'+name);if(!r.ok)throw Error('Fixture ausente: '+name);return r.arrayBuffer();}
function assert(ok,message){if(!ok)throw Error(message);}
function log(r){results.push(r);output.textContent=JSON.stringify({state:'running',results},null,2);}
try{
  const checks=await (await fetch('./verification/checks.json')).json();
  for(const c of checks){const data=new Float32Array(await bytes(c.input));const r=await predict(data,c.shape[2],c.shape[3]);const error=Math.abs(r.probability-c.expected[0]);assert(error<=1e-6+1e-5*Math.abs(c.expected[0]),'Diferencia ONNX/PyTorch');assert(r.predicted===(c.expected[0]>=.35?1:0),'Clase distinta');log({kind:'tensor',shape:c.shape,error});}
  const png=await (await fetch('./verification/png_checks.json')).json();
  for(const c of png.valid){const buffers=[];for(const file of c.files)buffers.push(await bytes(file));const p=await prepareTriplet(buffers);assert(await sha256(p.data.buffer)===c.sha256,'Píxeles distintos de Python');const r=await predict(p.data,p.height,p.width);const error=Math.abs(r.probability-c.expected);assert(error<=1e-6+1e-5*Math.abs(c.expected),'Predicción PNG distinta');log({kind:'PNG',shape:c.shape,interlaced:c.interlaced,pixels_exact:true,error});}
  for(const name of png.invalid){let rejected=false;try{await decodePNG(await bytes(name));}catch{rejected=true;}assert(rejected,'No rechazó '+name);log({kind:'invalid',name,rejected});}
  const first=await bytes(png.valid[0].files[0]);let rejected=false;try{await prepareTriplet([first,first,first]);}catch{rejected=true;}assert(rejected,'No rechazó fases duplicadas');log({kind:'duplicate',rejected});
  rejected=false;try{await prepareTriplet([first]);}catch{rejected=true;}assert(rejected,'No rechazó fases ausentes');log({kind:'missing',rejected});
  rejected=false;try{await prepareTriplet([first,await bytes(png.valid[1].files[1]),await bytes(png.valid[0].files[2])]);}catch{rejected=true;}assert(rejected,'No rechazó dimensiones distintas');log({kind:'shape_mismatch',rejected});
  output.textContent=JSON.stringify({state:'passed',results},null,2);
}catch(e){output.textContent=JSON.stringify({state:'failed',error:e.message,results},null,2);}
