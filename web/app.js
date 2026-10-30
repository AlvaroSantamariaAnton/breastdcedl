import {prepareTriplet,MAX_BYTES} from './png.js';
import {loadModel,predict} from './inference.js';
const $=id=>document.getElementById(id),inputs=['pre','early','late'].map($);
let report=null,busy=false,revision=0;
function clearResult(){window.dispatchEvent(new Event('prediction-cleared'));report=null;$('result').hidden=true;$('empty-result').hidden=false;$('enhancement').hidden=true;for(const id of ['pre','early','late'])$('preview-'+id).hidden=true;}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function clearInputErrors(){for(const input of inputs){input.removeAttribute('aria-invalid');input.removeAttribute('aria-describedby');}}
for(const input of inputs)input.addEventListener('change',()=>{revision++;clearInputErrors();clearResult();status('Imágenes cambiadas. Pulsa Analizar corte para obtener un nuevo resultado.');});
function draw(canvas,values,w,h,rgb=false){canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d'),im=ctx.createImageData(w,h);for(let k=0;k<w*h;k++){for(let c=0;c<3;c++)im.data[4*k+c]=rgb?values[3*k+c]:values[k];im.data[4*k+3]=255;}ctx.putImageData(im,0,0);canvas.hidden=false;}
$('reset').onclick=()=>{if(busy)return;revision++;inputs.forEach(i=>i.value='');clearInputErrors();clearResult();status('Selecciona las tres fases para comenzar.');};
$('analyze').onclick=async()=>{
  if(busy)return;clearInputErrors();clearResult();
  const missing=inputs.filter(input=>!input.files.length);
  if(missing.length){
    status('Falta alguna fase. Carga PRE, EARLY y LATE.',true);
    for(const input of missing){input.setAttribute('aria-invalid','true');input.setAttribute('aria-describedby','status');}
    missing[0].focus();return;
  }
  const rev=revision;busy=true;$('analyze').disabled=true;$('reset').disabled=true;inputs.forEach(i=>i.disabled=true);
  try{
    const files=inputs.map(i=>i.files[0]);
    if(files.some(f=>!f.size||f.size>MAX_BYTES))throw Error('Cada archivo debe contener un PNG de hasta 16 MiB.');
    status('Comprobando las tres fases…');const buffers=[];for(const f of files)buffers.push(await f.arrayBuffer());
    const prepared=await prepareTriplet(buffers),{width,height,phases,data}=prepared;
    phases.forEach((p,i)=>draw($('preview-'+['pre','early','late'][i]),p.pixels,width,height));
    const rgb=new Uint8Array(width*height*3);
    for(let k=0;k<width*height;k++){const delta=data[width*height+k]-data[k],mag=Math.abs(delta);rgb[3*k]=Math.round(255*(delta<0?1-mag:1));rgb[3*k+1]=Math.round(255*(1-mag));rgb[3*k+2]=Math.round(255*(delta>0?1-mag:1));}
    draw($('enhancement-canvas'),rgb,width,height,true);$('enhancement').hidden=false;
    status('Preparando el modelo en tu navegador… La primera carga puede tardar.');await loadModel();
    status('Analizando el corte…');await new Promise(r=>setTimeout(r,30));const start=performance.now();
    const result=await predict(data,height,width),seconds=(performance.now()-start)/1000;
    if(rev!==revision)return;
    report={...result,width,height,unit:'single_cut',device:'CPU · WebAssembly',inference_seconds:seconds,educational_only:true};
    $('probability').textContent=(result.probability*100).toLocaleString('es-ES',{minimumFractionDigits:2,maximumFractionDigits:2})+' %';
    $('meter-fill').style.width=(result.probability*100)+'%';$('class-result').textContent=result.predicted?'Predicción: pCR':'Predicción: no pCR';
    $('technical').textContent=`${width} × ${height} px · CPU (WebAssembly) · ${seconds.toFixed(2)} s de inferencia · Modelo SHA-256: ${result.checksum}`;
    window.dispatchEvent(new CustomEvent('prediction-ready',{detail:{prepared,result}}));
    $('empty-result').hidden=true;$('result').hidden=false;status('Análisis completado. Las imágenes permanecen en este navegador.');
  }catch(e){clearResult();status(e.message||'No se ha podido completar el análisis.',true);}
  finally{busy=false;$('analyze').disabled=false;$('reset').disabled=false;inputs.forEach(i=>i.disabled=false);}
};
$('download').onclick=()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='resultado-corte.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
