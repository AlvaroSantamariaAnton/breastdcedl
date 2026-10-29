import * as ort from './vendor/ort.wasm.min.mjs';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
const MODEL_URL='https://huggingface.co/AvaroSantamaria/breastdcedl/resolve/fa2209f4b4225ac639da8c6a0757448401e5046f/model.onnx';
let ready;
export async function sha256(buffer){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),v=>v.toString(16).padStart(2,'0')).join('');}
export async function loadModel(){
  if(!ready)ready=(async()=>{
    const mr=await fetch(new URL('./model/manifest.json',import.meta.url));if(!mr.ok)throw Error('No está disponible el modelo. Prepara primero los archivos de la web.');
    const manifest=await mr.json();
    if(manifest.threshold!==0.35 || !/^[a-f0-9]{64}$/.test(manifest.sha256))throw Error('El manifiesto no coincide con el predictor final aprobado.');
    const r=await fetch(MODEL_URL,{mode:'cors',credentials:'omit'});if(!r.ok)throw Error('No se ha podido descargar el modelo desde Hugging Face. Comprueba la conexión a Internet.');
    const bytes=await r.arrayBuffer();if(await sha256(bytes)!==manifest.sha256)throw Error('La huella del modelo no coincide.');
    const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
    return {session,manifest};
  })().catch(e=>{ready=null;throw e;});
  return ready;
}
export async function predict(data,height,width){
  if(!(data instanceof Float32Array)||data.length!==3*height*width||Math.min(height,width)<16||height*width>1048576||data.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('Tensor de entrada inválido.');
  const {session,manifest}=await loadModel();
  const tensor=new ort.Tensor('float32',data,[1,3,height,width]);
  let output;
  try{output=await session.run({images:tensor});const p=output.probability.data[0];if(!Number.isFinite(p)||p<0||p>1)throw Error('Salida no válida.');return {probability:p,predicted:p>=manifest.threshold?1:0,threshold:manifest.threshold,checksum:manifest.sha256};}
  finally{tensor.dispose();if(output)Object.values(output).forEach(t=>t.dispose());}
}

let explanationReady;
export async function explain(data,height,width){
  if(!(data instanceof Float32Array)||data.length!==3*height*width||Math.min(height,width)<16||height*width>1048576||data.some(x=>!Number.isFinite(x)||x<0||x>1))throw Error('Entrada Grad-CAM inválida.');
  if(!explanationReady)explanationReady=(async()=>{
    const r=await fetch(new URL('./model/gradcam.json',import.meta.url));
    if(!r.ok)throw Error('Falta preparar el manifiesto Grad-CAM.');
    const manifest=await r.json();
    if(!/^[a-f0-9]{64}$/.test(manifest.sha256)||!manifest.url.startsWith('https://huggingface.co/AvaroSantamaria/breastdcedl/resolve/'))throw Error('Manifiesto Grad-CAM inválido.');
    const model=await fetch(manifest.url,{mode:'cors',credentials:'omit'});
    if(!model.ok)throw Error('Grad-CAM aún no está disponible en Hugging Face. Debe subirse model-gradcam.onnx.');
    const bytes=await model.arrayBuffer();if(await sha256(bytes)!==manifest.sha256)throw Error('La huella de Grad-CAM no coincide con la exportación comprobada.');
    return {session:await ort.InferenceSession.create(bytes,{executionProviders:['wasm']}),manifest};
  })().catch(e=>{explanationReady=null;throw e;});
  const {session,manifest}=await explanationReady;
  const tensor=new ort.Tensor('float32',data,[1,3,height,width]);let outputs;
  try{
    outputs=await session.run({images:tensor});
    const maps=[outputs.cam_no_pcr,outputs.cam_pcr];
    if(maps.some(t=>t.dims.length!==3||t.dims[0]!==1||t.dims[1]!==Math.floor(height/16)||t.dims[2]!==Math.floor(width/16)||t.data.some(v=>!Number.isFinite(v)||v<0)))throw Error('Mapa Grad-CAM inválido.');
    const probability=outputs.probability.data[0];if(!Number.isFinite(probability)||probability<0||probability>1)throw Error('Probabilidad Grad-CAM inválida.');
    return {probability,maps:maps.map(t=>Float32Array.from(t.data)),height:maps[0].dims[1],width:maps[0].dims[2],checksum:manifest.sha256};
  }finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(t=>t.dispose());}
}
