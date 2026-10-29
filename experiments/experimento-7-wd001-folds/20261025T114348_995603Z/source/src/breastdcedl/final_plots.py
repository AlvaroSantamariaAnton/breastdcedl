"""Figuras descriptivas del ajuste final, sin fingir validación."""
from .artifacts import prepare_plotting, style_axes, confusion_report


def plot_final(folder, state):
    prepare_plotting()
    import matplotlib.pyplot as plt
    target = folder / 'figures'
    target.mkdir(exist_ok=True)
    rows = state['history']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes[:2], ('loss', 'auc'), ('Pérdida BCE', 'ROC-AUC por paciente')):
        ax.plot([r['epoch'] for r in rows], [r['train_' + key] for r in rows], color='#2c7ddd', label='entrenamiento')
        ax.set(title=title, xlabel='Época', ylim=(0, max(1, max(r['train_' + key] for r in rows)) * 1.03))
        style_axes(ax)
        ax.legend()
    roc = state['roc']['train']
    axes[2].plot(roc['fpr'], roc['tpr'], color='#2c7ddd')
    axes[2].plot([0, 1], [0, 1], ':', color='gray')
    axes[2].set(title='ROC entrenamiento', xlabel='Tasa de falsos positivos', ylabel='Sensibilidad', xlim=(0, 1), ylim=(0, 1))
    style_axes(axes[2])
    fig.suptitle('Ajuste final · solo entrenamiento · sin validación ni test')
    fig.tight_layout()
    fig.savefig(target / 'curves.png', dpi=150)
    plt.close(fig)
    counts = {k: rows[-1]['train_' + k] for k in ('tn', 'fp', 'fn', 'tp')}
    confusion_report(target / 'confusion_train.png', counts, 'Entrenamiento · no evaluación independiente', state['config']['evaluation']['threshold'])
