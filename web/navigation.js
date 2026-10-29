const links=[...document.querySelectorAll('[data-page]')];
const sections=[...document.querySelectorAll('.page')];
function navigate(event){
  const requested=location.hash.slice(1)||'inicio';
  const target=sections.find(s=>s.id===requested)?requested:'inicio';
  for(const section of sections)section.hidden=section.id!==target;
  for(const link of links){if(link.dataset.page===target)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
  const heading=document.querySelector('#'+target+' h1');
  document.title=(heading?.textContent.trim()||'Inicio')+' · BreastDCE';
  window.scrollTo({top:0,behavior:'instant'});
  if(event && heading){heading.tabIndex=-1;heading.focus({preventScroll:true});}
  const active=links.find(link=>link.dataset.page===target);
  if(active && window.matchMedia('(max-width:800px)').matches){const nav=active.parentElement;nav.scrollLeft=active.offsetLeft-nav.offsetLeft-(nav.clientWidth-active.offsetWidth)/2;}
}
window.addEventListener('hashchange',navigate);
navigate();

document.querySelector(".skip-link")?.addEventListener("click",event=>{event.preventDefault();document.getElementById("main-content").focus();});
