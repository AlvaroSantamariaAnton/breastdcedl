// PNG gris de 8 bits: decodificar los valores originales, sin gestión de color de canvas.
export const MAX_BYTES = 16 * 1024 * 1024;
export const MAX_PIXELS = 1024 * 1024;
const crcTable = Uint32Array.from({length:256}, (_,n) => {
  for(let k=0;k<8;k++) n=(n&1)?0xedb88320^(n>>>1):n>>>1;
  return n>>>0;
});
function crc(bytes) { let c=0xffffffff; for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8); return (c^0xffffffff)>>>0; }
function paeth(a,b,c){const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);return pa<=pb&&pa<=pc?a:pb<=pc?b:c;}
export async function decodePNG(buffer){
  const b=new Uint8Array(buffer),v=new DataView(buffer);
  if(b.length<33 || b.length>MAX_BYTES || ![137,80,78,71,13,10,26,10].every((x,i)=>b[i]===x))throw Error('Se requiere un PNG válido de hasta 16 MiB.');
  let pos=8,width,height,interlace,ended=false,seenData=false,closedData=false;const chunks=[];
  while(pos+12<=b.length){
    const n=v.getUint32(pos),end=pos+12+n;
    if(end>b.length)throw Error('PNG truncado.');
    const type=String.fromCharCode(...b.subarray(pos+4,pos+8));
    if(crc(b.subarray(pos+4,pos+8+n))!==v.getUint32(pos+8+n))throw Error('PNG corrupto: checksum incorrecto.');
    if(pos===8&&type!=='IHDR')throw Error('Cabecera PNG ausente.');
    if(type==='IHDR'){
      if(pos!==8||n!==13)throw Error('Cabecera PNG inválida.');
      width=v.getUint32(pos+8);height=v.getUint32(pos+12);interlace=b[pos+20];
      if(width<16||height<16||width*height>MAX_PIXELS)throw Error('Mínimo 16 píxeles por lado y máximo 1.048.576 píxeles por fase.');
      if(b[pos+16]!==8||b[pos+17]!==0||b[pos+18]!==0||b[pos+19]!==0||interlace>1)throw Error('Usa PNG gris de 8 bits; no RGB, paleta ni 16 bits.');
    }else if(type==='IDAT'){
      if(closedData)throw Error('Bloques PNG desordenados.');
      seenData=true;chunks.push(b.slice(pos+8,pos+8+n));
    }else{
      if(seenData)closedData=true;
      if(type==='acTL')throw Error('No se admiten PNG animados.');
      if(type==='IEND'){if(n!==0||!seenData||end!==b.length)throw Error('Final PNG inválido.');ended=true;break;}
      if(type[0]===type[0].toUpperCase()&&type!=='PLTE')throw Error('Bloque PNG no compatible.');
    }
    pos=end;
  }
  if(!ended)throw Error('PNG incompleto.');
  const passes=interlace?[[0,0,8,8],[4,0,8,8],[0,4,4,8],[2,0,4,4],[0,2,2,4],[1,0,2,2],[0,1,1,2]]:[[0,0,1,1]];
  const size=(start,step,total)=>Math.max(0,Math.ceil((total-start)/step));
  const expected=passes.reduce((s,[x,y,dx,dy])=>{const w=size(x,dx,width),h=size(y,dy,height);return s+(w&&h?(w+1)*h:0);},0);
  const reader=new Blob(chunks).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
  const raw=new Uint8Array(expected);let offset=0;
  try{while(true){const {value,done}=await reader.read();if(done)break;if(offset+value.length>expected)throw Error('Datos PNG excesivos.');raw.set(value,offset);offset+=value.length;}}finally{await reader.cancel();}
  if(offset!==expected)throw Error('Datos PNG incompletos.');
  const pixels=new Uint8Array(width*height);offset=0;
  for(const [sx,sy,dx,dy] of passes){
    const w=size(sx,dx,width),h=size(sy,dy,height);if(!w||!h)continue;
    let previous=new Uint8Array(w);
    for(let y=0;y<h;y++){
      const filter=raw[offset++],row=new Uint8Array(w);if(filter>4)throw Error('Filtro PNG inválido.');
      for(let x=0;x<w;x++){
        const a=x?row[x-1]:0,up=previous[x],c=x?previous[x-1]:0;
        const add=[0,a,up,Math.floor((a+up)/2),paeth(a,up,c)][filter];
        row[x]=(raw[offset++]+add)&255;pixels[(sy+y*dy)*width+sx+x*dx]=row[x];
      }previous=row;
    }
  }
  return {width,height,pixels};
}
export async function prepareTriplet(buffers){
  if(buffers.length!==3)throw Error('Carga PRE, EARLY y LATE del mismo corte.');
  const phases=[];for(const b of buffers)phases.push(await decodePNG(b));
  const {width,height}=phases[0];
  if(phases.some(p=>p.width!==width||p.height!==height))throw Error('Las tres fases deben tener el mismo tamaño.');
  for(let i=0;i<3;i++)for(let j=0;j<i;j++)if(phases[i].pixels.every((v,k)=>v===phases[j].pixels[k]))throw Error('Hay fases duplicadas: contienen los mismos píxeles.');
  const data=new Float32Array(3*width*height);
  phases.forEach((p,i)=>p.pixels.forEach((v,k)=>{data[i*width*height+k]=v/255;}));
  return {data,phases,width,height};
}
