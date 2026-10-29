import os, tempfile
from pathlib import Path
folder=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="calibration-mpl-") as cache:
    os.environ['MPLCONFIGDIR']=cache
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    bins=pd.read_csv(folder/'reliability_bins.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    g=bins[bins.scope=='total']
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Calibración ideal')
    axes[0].errorbar(g.mean_probability,g.observed,yerr=[g.observed-g.wilson_low,g.wilson_high-g.observed],fmt='o-',capsize=4,label='Normal · validación OOF')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Probabilidad media predicha',ylabel='Proporción observada de pCR',title='Fiabilidad · intervalos descriptivos del 95%')
    axes[0].legend(fontsize=8)
    axes[1].bar(g.bin_lower+.05,g.n,width=.085,color='#2c7ddd')
    for _,r in g.iterrows():axes[1].text(r.bin_lower+.05,r.n+5,str(int(r.n)),ha='center',fontsize=9)
    axes[1].set(xlim=(0,1),xlabel='Probabilidad predicha · intervalos de 0,10',ylabel='Pacientes',title='Dónde se concentran las probabilidades')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Candidata normal · 1.097 pacientes de validación · sin ajuste de calibración')
    fig.savefig(folder/'calibration.png',dpi=160)
    plt.close(fig)
