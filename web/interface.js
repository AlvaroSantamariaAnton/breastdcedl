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
// Tema claro u oscuro: sigue al sistema hasta que se elige uno; solo se guarda esa preferencia.
const root=document.documentElement,themeToggle=document.getElementById('theme-toggle');
const isDark=()=>root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
function syncTheme(){const dark=isDark();themeToggle.setAttribute('aria-pressed',String(dark));}
themeToggle.onclick=()=>{const next=isDark()?'light':'dark';root.dataset.theme=next;try{localStorage.setItem('breastdce-tema',next);}catch{}syncTheme();};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',syncTheme);syncTheme();
// Sombra de la cabecera al desplazar.
const header=document.querySelector('.project-header');
function syncHeader(){header.classList.toggle('scrolled',scrollY>8);}
addEventListener('scroll',syncHeader,{passive:true});syncHeader();
// Aparición suave de los bloques al entrar en pantalla (sin efecto si se prefiere menos movimiento).
if('IntersectionObserver' in window&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
 const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('visible');observer.unobserve(entry.target);}},{rootMargin:'0px 0px -6% 0px'});
 for(const block of document.querySelectorAll('.page > :not(.page-heading):not(.hero), .cards > *, .two-col > *')){
  if(block.matches('.cards, .two-col'))continue;
  block.classList.add('reveal');observer.observe(block);
 }
}
// Arrastrar un PNG sobre una fase equivale a seleccionarlo: se dispara el mismo evento change.
for(const zone of document.querySelectorAll('.upload')){
 const input=zone.querySelector('input[type=file]');
 for(const type of ['dragenter','dragover'])zone.addEventListener(type,event=>{if(input.disabled)return;event.preventDefault();zone.classList.add('dragover');});
 zone.addEventListener('dragleave',event=>{if(!zone.contains(event.relatedTarget))zone.classList.remove('dragover');});
 zone.addEventListener('drop',event=>{
  zone.classList.remove('dragover');const file=event.dataTransfer?.files?.[0];
  if(input.disabled||!file)return;
  event.preventDefault();const transfer=new DataTransfer();transfer.items.add(file);
  input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));
 });
}
image.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggleZoom();}});
