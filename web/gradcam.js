import {explain} from './inference.js';
const $=id=>document.getElementById(id);
let current=null,map=null,revision=0;
window.addEventListener('prediction-cleared',()=>{revision++;current=null;map=null;$('gradcam').hidden=true;$('gradcam-view').hidden=true;$('gradcam-status').textContent='';$('gradcam-run').disabled=false;});
window.addEventListener('prediction-ready',event=>{current=event.detail;map=null;$('gradcam').hidden=false;$('gradcam-target').value=String(current.result.predicted);});
function render(){
  if(!current||!map)return;
  const target=Number($('gradcam-target').value),phase=['PRE','EARLY','LATE'][Number($('gradcam-phase').value)];
  $('gradcam-class-label').textContent=`Clase explicada: ${target?'pCR (1)':'no pCR (0)'}`;
  $('gradcam-background-label').textContent=`Fondo: ${phase}. Cambiar la fase solo cambia el fondo, no el mapa.`;
  $('gradcam-decision-label').textContent=`Predicción original: ${current.result.predicted?'pCR (1)':'no pCR (0)'} · umbral 0,35.${target!==current.result.predicted?' Estás viendo la clase alternativa; la predicción no ha cambiado.':''}`;
  const {width:w,height:h,phases}=current.prepared,values=map.maps[Number($('gradcam-target').value)];
  let max=0,min=Infinity;for(const v of values){max=Math.max(max,v);min=Math.min(min,v);}
  const spatial=map.width>1&&map.height>1&&max>1e-8&&(max-min)>max*1e-6;
  const canvas=$('gradcam-canvas');canvas.width=w;canvas.height=h;
  const ctx=canvas.getContext('2d'),im=ctx.createImageData(w,h),pixels=phases[Number($('gradcam-phase').value)].pixels;
  const opacity=Number($('gradcam-opacity').value)/100;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const u=Math.max(0,Math.min(map.width-1,(x+.5)*map.width/w-.5)),v=Math.max(0,Math.min(map.height-1,(y+.5)*map.height/h-.5));
    const x0=Math.floor(u),y0=Math.floor(v),x1=Math.min(x0+1,map.width-1),y1=Math.min(y0+1,map.height-1),dx=u-x0,dy=v-y0;
    const value=(values[y0*map.width+x0]*(1-dx)+values[y0*map.width+x1]*dx)*(1-dy)+(values[y1*map.width+x0]*(1-dx)+values[y1*map.width+x1]*dx)*dy;
    const t=spatial?value/max:0,alpha=t*opacity,k=y*w+x;
    const color=[255,Math.round(255*(1-t)),0];
    for(let c=0;c<3;c++)im.data[4*k+c]=Math.round(pixels[k]*(1-alpha)+color[c]*alpha);
    im.data[4*k+3]=255;
  }
  ctx.putImageData(im,0,0);
  $('gradcam-note').textContent=spatial?`Mapa para ${$('gradcam-target').value==='1'?'pCR (1)':'no pCR (0)'} · rejilla original ${map.width} × ${map.height}, ampliada para mostrarla. Mayor color indica mayor intensidad relativa, no certeza clínica.`:'No hay contraste espacial suficiente para localizar regiones en este mapa. Se muestra la imagen sin colorear.';
}
$('gradcam-run').onclick=async()=>{
  if(!current)return;const rev=revision,sample=current;$('gradcam-run').disabled=true;$('gradcam-status').classList.remove('error');$('gradcam-status').textContent='Descargando el modelo de explicación y calculando Grad-CAM…';
  try{
    const p=sample.prepared,result=await explain(p.data,p.height,p.width);if(rev!==revision)return;
    if(Math.abs(result.probability-sample.result.probability)>1e-6+1e-5*Math.abs(sample.result.probability))throw Error('La predicción del modelo de explicación no coincide; se oculta el mapa.');
    map=result;$('gradcam-view').hidden=false;render();$('gradcam-status').textContent='Mapa calculado. La predicción original se mantiene.';
  }catch(e){if(rev===revision){map=null;$('gradcam-view').hidden=true;$('gradcam-status').textContent=e.message;$('gradcam-status').classList.add('error');}}
  finally{if(rev===revision)$('gradcam-run').disabled=false;}
};
for(const id of ['gradcam-target','gradcam-phase','gradcam-opacity'])$(id).addEventListener('input',render);
