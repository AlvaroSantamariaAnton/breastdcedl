"""Crear PNG sintéticos de prueba y referencias Python; nunca abre dataset/test."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from breastdcedl.input import decode_triplet
from breastdcedl.inference import FoldEnsemble


def chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))


def png(array,interlace=False):
    h,w=array.shape
    passes=[(0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2)] if interlace else [(0,0,1,1)]
    raw=bytearray()
    for sx,sy,dx,dy in passes:
        sub=array[sy::dy,sx::dx]
        if not sub.size:continue
        previous=np.zeros(sub.shape[1],dtype=np.uint8)
        for y,row in enumerate(sub):
            f=y%5;raw.append(f)
            for x,value in enumerate(row):
                a=int(row[x-1]) if x else 0;b=int(previous[x]);c=int(previous[x-1]) if x else 0
                p=a+b-c;pa,pb,pc=abs(p-a),abs(p-b),abs(p-c)
                paeth=a if pa<=pb and pa<=pc else b if pb<=pc else c
                raw.append((int(value)-[0,a,b,(a+b)//2,paeth][f])&255)
            previous=row
    header=struct.pack('>IIBBBBB',w,h,8,0,0,0,int(interlace))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'gAMA',struct.pack('>I',45455))+chunk(b'IDAT',zlib.compress(bytes(raw)))+chunk(b'IEND',b'')


def main():
    out=ROOT/'web/verification';out.mkdir(exist_ok=True)
    model=FoldEnsemble(ROOT/'experiments/modelo_final.json',ROOT)
    checks=[]
    for h,w,interlace in [(256,256,False),(244,244,False),(31,49,True),(16,16,True)]:
        files=[];phases={}
        for i,phase in enumerate(('PRE','EARLY','LATE')):
            yy,xx=np.indices((h,w));pixels=((xx*3+yy*5+i*61)%256).astype('uint8')
            data=png(pixels,interlace);name=f'{h}x{w}_{phase}.png';(out/name).write_bytes(data)
            files.append(name);phases[phase]=data
        tensor=decode_triplet(phases)
        checks.append(dict(files=files,shape=list(tensor.shape),sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                           expected=model.predict_patient([tensor]),interlaced=interlace))
    invalid=[]
    for mode in ['RGB','I;16']:
        buf=io.BytesIO();Image.new(mode,(32,32)).save(buf,format='PNG');name=f'invalid_{mode.replace(";", "")}.png';(out/name).write_bytes(buf.getvalue());invalid.append(name)
    bad=bytearray((out/checks[0]['files'][0]).read_bytes());bad[50]^=1
    (out/'invalid_crc.png').write_bytes(bad);invalid.append('invalid_crc.png')
    (out/'png_checks.json').write_text(json.dumps({'valid':checks,'invalid':invalid},indent=2),encoding='utf-8')
    print('Referencias PNG sintéticas preparadas, sin dataset.')


if __name__=='__main__':main()
