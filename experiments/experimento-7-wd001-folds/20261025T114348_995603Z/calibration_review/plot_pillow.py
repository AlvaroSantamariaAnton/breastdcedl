import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
folder=Path(__file__).resolve().parent
rows=[{k:(v if k=='scope' else float(v)) for k,v in r.items()} for r in csv.DictReader((folder/'reliability_bins.csv').open()) if r['scope']=='total']
im=Image.new('RGB',(1600,720),'white');d=ImageDraw.Draw(im)
def font(n):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
def text(x,y,s,n=20,fill='#263238'):d.text((x,y),s,font=font(n),fill=fill)
text(80,24,'Calibración de la candidata normal · validación OOF · 1.097 pacientes',30)
text(80,68,'Sin ajustar probabilidades ni utilizar test',20)
left,top,w,h=110,160,610,430
X=lambda v:left+v*w
Y=lambda v:top+h-v*h
for i in range(6):
 v=i/5;d.line((X(v),top,X(v),top+h),fill='#e6e6e6');d.line((left,Y(v),left+w,Y(v)),fill='#e6e6e6');text(X(v)-15,top+h+12,f'{v:.1f}',18);text(left-48,Y(v)-10,f'{v:.1f}',18)
d.line((left,top,left,top+h,left+w,top+h),fill='black',width=2)
d.line((X(0),Y(0),X(1),Y(1)),fill='#999999',width=2)
points=[(X(r['mean_probability']),Y(r['observed'])) for r in rows]
d.line(points,fill='#2869a6',width=3)
for r,(x,y) in zip(rows,points):
 lo,hi=Y(r['wilson_low']),Y(r['wilson_high']);d.line((x,lo,x,hi),fill='#2869a6',width=2)
 for yy in [lo,hi]:d.line((x-7,yy,x+7,yy),fill='#2869a6',width=2)
 d.ellipse((x-5,y-5,x+5,y+5),fill='#2869a6')
text(110,120,'Proporción real de pCR',22);text(180,640,'Probabilidad media predicha',22)
left2=900;w2=590
text(left2,120,'Pacientes por intervalo de probabilidad',22)
for tick in [0,200,400,600]:
 y=top+h-tick/650*h;d.line((left2,y,left2+w2,y),fill='#e6e6e6');text(left2-52,y-10,str(tick),18)
for r in rows:
 x=left2+(r['bin_lower']+.05)*w2;ww=.08*w2;y=top+h-r['n']/650*h
 d.rectangle((x-ww/2,y,x+ww/2,top+h),fill='#2869a6');text(x-15,y-28,str(int(r['n'])),18)
for i in range(6):text(left2+i/5*w2-15,top+h+12,f'{i/5:.1f}',18)
text(left2+55,640,'Probabilidad predicha (intervalos de 0,10)',20)
text(110,690,'Diagonal gris: calibración ideal. Barras: IC95% Wilson descriptivo; pocos casos en los extremos.',18)
im.save(folder/'calibration.png')
