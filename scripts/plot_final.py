"""Figuras del predictor final desde CSV/JSON guardados. Sin redes ni inferencia."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'experiments/modelo_final.json').read_text(encoding='utf-8'))
    run = ROOT/Path(manifest['members'][0]['checkpoint']).parents[2]
    out = run/'final_test'
    sources = [out/'metrics.json',out/'patient_scores.csv',ROOT/'experiments/modelo_final.json']
    sources += [run/f'fold_{m["fold"]}/metrics.csv' for m in manifest['members']]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    metrics = json.loads((out/'metrics.json').read_text())
    scores = pd.read_csv(out/'patient_scores.csv')
    assert len(scores)==176 and scores.patient_id.nunique()==176
    assert ((scores.probability>=manifest['threshold']).astype(int)==scores.predicted).all()
    with tempfile.TemporaryDirectory(prefix='plot_final_cache_',dir=ROOT) as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                             'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,
                             'grid.alpha':.2,'figure.facecolor':'white'})
        blue,orange='#2575bc','#e66c35'
        fig,axes=plt.subplots(5,2,figsize=(15,17),layout='constrained')
        chosen=[]
        for member,ax in zip(manifest['members'],axes):
            frame=pd.read_csv(run/f'fold_{member["fold"]}/metrics.csv')
            row=frame[frame.epoch==member['epoch']].iloc[0];chosen.append(row)
            for a,key,label in [(ax[0],'loss','BCE por corte'),(ax[1],'auc','ROC-AUC por paciente')]:
                a.plot(frame.epoch,frame['train_'+key],color=blue,label='Entrenamiento')
                a.plot(frame.epoch,frame['val_'+key],color=orange,label='Validación')
                a.axvline(member['epoch'],color='#555',ls=':',label=f'Época elegida: {member["epoch"]}')
                a.scatter([member['epoch']],[row['val_'+key]],color=orange,zorder=4)
                a.set(xlabel='Época',ylabel=label,xlim=(1,50))
                a.legend(fontsize=9,loc='best')
            ax[0].set_title(f'Fold {member["fold"]} · pérdida (eje Y propio)')
            ax[1].set_ylim(.35,.9)
            ax[1].axhline(.5,color='#999',ls='--',lw=1)
            ax[1].set_title(f'Fold {member["fold"]} · AUC · gap elegido {row.train_auc-row.val_auc:+.3f}')
        fig.suptitle('Modelo final · aprendizaje de las cinco redes conservadas\nCurvas históricas train/validación; no entrenamiento nuevo ni curvas de test por época',fontsize=17)
        fig.savefig(out/'curvas_aprendizaje_folds.png',dpi=150);plt.close(fig)

        selected=pd.DataFrame(chosen)
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
        means=[selected.train_auc.mean(),selected.val_auc.mean()]
        std=[selected.train_auc.std(ddof=1),selected.val_auc.std(ddof=1)]
        ax.bar([0,1],means,yerr=std,capsize=8,color=[blue,orange],width=.55)
        for i,col in enumerate(['train_auc','val_auc']):
            ax.scatter(np.full(5,i)+np.linspace(-.08,.08,5),selected[col],color='#16324f',s=28,zorder=4)
            ax.text(i,means[i]+std[i]+.035,f'{means[i]:.3f} ± {std[i]:.3f}',ha='center',weight='bold')
        ax.axhline(.5,color='#999',ls='--');ax.set(xticks=[0,1],xticklabels=['Entrenamiento','Validación'],ylim=(0,1),ylabel='ROC-AUC por paciente')
        ax.set_title('AUC en la época elegida · media ± DE muestral\nCinco modelos individuales; no AUC del ensemble en test')
        fig.savefig(out/'auc_epocas_elegidas.png',dpi=180);plt.close(fig)

        total=metrics['overall'];fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained')
        ax=axes[0,0];roc=total['roc'];ax.plot(roc['fpr'],roc['tpr'],color=blue,lw=2,label=f'AUC = {total["auc"]:.3f}')
        ax.plot([0,1],[0,1],'--',color='#999',label='Referencia AUC 0,50')
        ax.scatter([1-total['recall'][0]],[total['recall'][1]],color=orange,zorder=5,label='Umbral 0,35')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − especificidad',ylabel='Sensibilidad pCR',title=f'ROC global · IC95% [{total["auc_ci95"][0]:.3f}, {total["auc_ci95"][1]:.3f}]');ax.legend(loc='lower right')
        ax=axes[0,1]
        for cohort,label,color in [('duke','Duke','#258f70'),('spy1','I-SPY1','#b85d87'),('spy2','I-SPY2',blue)]:
            result=metrics['cohorts'][cohort];r=result['roc']
            ax.plot(r['fpr'],r['tpr'],color=color,label=f'{label}: AUC {result["auc"]:.3f} · n={result["patients"]}')
        ax.plot([0,1],[0,1],'--',color='#999');ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − especificidad',ylabel='Sensibilidad pCR',title='ROC por cohorte · análisis descriptivo');ax.legend(loc='lower right')
        ax=axes[1,0];bins=total['calibration_bins']
        ax.plot([0,1],[0,1],'--',color='#999',label='Calibración ideal')
        xs=[b['mean_probability'] for b in bins];ys=[b['observed'] for b in bins]
        ax.plot(xs,ys,color=blue,marker='o',label='Bins fijos de anchura 0,10')
        for b in bins:ax.annotate(f'n={b["n"]}',(b['mean_probability'],b['observed']),xytext=(7,-14 if b['observed']>.9 else 8),textcoords='offset points',fontsize=9)
        ax.set(xlim=(0,1),ylim=(-.04,1.06),xlabel='Probabilidad media predicha',ylabel='Proporción observada de pCR',title=f'Calibración descriptiva · Brier {total["brier"]:.3f}');ax.legend(loc='lower right')
        ax=axes[1,1]
        for label,name,color in [(0,'No pCR real (n=123)',blue),(1,'pCR real (n=53)',orange)]:
            ax.hist(scores.loc[scores.label==label,'probability'],bins=np.linspace(0,1,21),histtype='step',lw=2,color=color,label=name)
        ax.axvline(manifest['threshold'],color='#444',ls='--',label='Umbral fijado: 0,35')
        ax.set(xlim=(0,1),xlabel='Probabilidad de pCR por paciente',ylabel='Número de pacientes',title='Distribución de probabilidades · sin normalizar recuentos');ax.legend()
        fig.suptitle('Test final · 176 pacientes · conjunto de cinco redes · umbral 0,35\nFiguras desde predicciones guardadas; sin repetir inferencia ni ajustar el modelo',fontsize=17)
        fig.savefig(out/'curvas_test.png',dpi=170);plt.close(fig)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/'figures_audit.json').write_text(json.dumps({'sources':hashes,'inference_repeated':False,'training_performed':False,
        'figures':['curvas_aprendizaje_folds.png','auc_epocas_elegidas.png','curvas_test.png'],
        'note':'No test threshold sweep; no new confidence interval estimation; historical learning curves distinguished from ensemble test.'},indent=2),encoding='utf-8')
    print(f'Tres figuras guardadas en {out}')



def plot_learning_comparison():
    """Solo métricas históricas de desarrollo; no lee ni recalcula test."""
    import shutil
    manifest = json.loads((ROOT/'experiments/modelo_final.json').read_text(encoding='utf-8'))
    run = ROOT/Path(manifest['members'][0]['checkpoint']).parents[2]
    runs = {'normal': run, 'ponderada': run/'comparison_weighted'}
    with tempfile.TemporaryDirectory(prefix='plot_learning_',dir=ROOT) as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                             'axes.spines.top':False,'axes.spines.right':False,
                             'axes.grid':True,'grid.alpha':.2})
        for label, run in runs.items():
            files = [run/f'fold_{i}/metrics.csv' for i in range(5)]
            hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            frames = [pd.read_csv(p) for p in files]
            chosen = [f.loc[f.val_auc.idxmax()] for f in frames]
            blue, orange = '#2575bc', '#e66c35'
            fig, axes = plt.subplots(1,3,figsize=(19,5.5),layout='constrained')
            for i,(frame,row) in enumerate(zip(frames,chosen)):
                for ax,key in zip(axes[:2],['loss','auc']):
                    ax.plot(frame.epoch,frame['train_'+key],color=blue,alpha=.75,label='Entrenamiento' if i==0 else None)
                    ax.plot(frame.epoch,frame['val_'+key],color=orange,alpha=.85,label='Validación' if i==0 else None)
                    ax.scatter(row.epoch,row['val_'+key],color=orange,edgecolor='white',zorder=4,label='Época elegida' if i==0 else None)
                    ax.set(xlabel='Época',xlim=(1,50))
            axes[0].set(title='Pérdida · cinco folds superpuestos',ylabel='BCE por corte (escala propia)')
            axes[1].set(title='AUC · cinco folds superpuestos',ylabel='ROC-AUC por paciente',ylim=(.35,.9))
            axes[1].axhline(.5,color='#999',ls='--')
            for ax in axes[:2]: ax.legend(fontsize=9)
            vals = np.array([[r.train_auc,r.val_auc] for r in chosen])
            mean,sd = vals.mean(axis=0),vals.std(axis=0,ddof=1)
            axes[2].bar([0,1],mean,yerr=sd,capsize=6,color=[blue,orange],width=.6)
            for j in range(2):
                axes[2].scatter(np.full(5,j)+np.linspace(-.06,.06,5),vals[:,j],color='#16324f',zorder=4,s=22)
                axes[2].text(j,mean[j]+sd[j]+.03,f'{mean[j]:.3f} ± {sd[j]:.3f}',ha='center',weight='bold')
            axes[2].set(xticks=[0,1],xticklabels=['Entrenamiento','Validación'],ylim=(0,1),ylabel='ROC-AUC por paciente',title='Época elegida · media ± DE muestral')
            axes[2].axhline(.5,color='#999',ls='--')
            fig.suptitle(f'BCE {label} · weight decay 0,001 · cinco folds terminados\nVista conjunta de modelos individuales; no curva de entrenamiento del ensemble',fontsize=15)
            combined=run/f'curvas_conjuntas_{label}.png'
            fig.savefig(combined,dpi=150);plt.close(fig)
            fig,axes=plt.subplots(5,2,figsize=(15,17),layout='constrained')
            for i,(frame,row) in enumerate(zip(frames,chosen)):
                for ax,key in zip(axes[i],['loss','auc']):
                    ax.plot(frame.epoch,frame['train_'+key],color=blue,label='Entrenamiento')
                    ax.plot(frame.epoch,frame['val_'+key],color=orange,label='Validación')
                    ax.axvline(row.epoch,color='#666',ls=':',label=f'Época elegida: {int(row.epoch)}')
                    ax.scatter(row.epoch,row['val_'+key],color=orange,zorder=4)
                    ax.set(xlabel='Época',xlim=(1,50));ax.legend(fontsize=9)
                axes[i,0].set(title=f'Fold {i} · pérdida (escala propia)',ylabel='BCE por corte')
                axes[i,1].set(title=f'Fold {i} · AUC · diferencia elegida {row.train_auc-row.val_auc:+.3f}',ylabel='ROC-AUC por paciente',ylim=(.35,.9))
                axes[i,1].axhline(.5,color='#999',ls='--')
            fig.suptitle(f'BCE {label} · detalle de los cinco folds · weight decay 0,001\nCurvas históricas; puntos y líneas verticales indican las épocas elegidas',fontsize=17)
            detail=run/f'curvas_detalle_{label}.png'
            fig.savefig(detail,dpi=150);plt.close(fig)
            for source in [combined,detail]: shutil.copyfile(source,ROOT/'web/assets'/source.name)
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            (run/'learning_figures_audit.json').write_text(json.dumps({'sources':hashes,'selected_epochs':[int(r.epoch) for r in chosen],'training_performed':False,'inference_repeated':False,'figures':[combined.name,detail.name]},indent=2),encoding='utf-8')
            print(f'Figuras {label}: {combined.name}, {detail.name}')

if __name__=='__main__':
    import sys
    if '--learning-comparison' in sys.argv: plot_learning_comparison()
    else: main()

