const viewer=document.getElementById('figure-viewer'),image=document.getElementById('viewer-image');
let origin;
for(const link of document.querySelectorAll('.figure a, .architecture-figure a')){
 if(!/\.(png|svg)(?:\?|$)/i.test(link.getAttribute('href')||''))continue;
 link.addEventListener('click',event=>{
  if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||typeof viewer.showModal!=='function')return;
  event.preventDefault();origin=link;
  const description=link.querySelector('img')?.alt||'Arquitectura de la CNN';
  image.src=link.href;image.alt=description;document.getElementById('viewer-title').textContent=description;
  viewer.querySelector('.viewer-body').classList.remove('zoomed');image.setAttribute('aria-pressed','false');viewer.showModal();
 });
}
document.getElementById('viewer-close').onclick=()=>viewer.close();
viewer.addEventListener('click',event=>{if(event.target===viewer){const r=viewer.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)viewer.close();}});
viewer.addEventListener('close',()=>origin?.focus());
function toggleZoom(){const zoomed=viewer.querySelector('.viewer-body').classList.toggle('zoomed');image.setAttribute('aria-pressed',String(zoomed));}
image.onclick=toggleZoom;
image.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggleZoom();}});
