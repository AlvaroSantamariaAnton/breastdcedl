"""Archivos atómicos y gráficas dentro de la carpeta local de cada ejecución."""
import csv
import io
import json
import os
import tempfile
from pathlib import Path

_plot_cache = None


def prepare_plotting():
    """Caché efímera fuera del run; TemporaryDirectory la limpia al salir."""
    global _plot_cache
    if _plot_cache is None:
        _plot_cache = tempfile.TemporaryDirectory(prefix='breastdcedl-mpl-', ignore_cleanup_errors=True)
        os.environ['MPLCONFIGDIR'] = _plot_cache.name
    import matplotlib
    matplotlib.use('Agg')


def style_axes(ax):
    from matplotlib.ticker import FuncFormatter
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c6c6bf')
    ax.tick_params(colors='#606060', labelsize=9)
    ax.xaxis.label.set_color('#606060')
    ax.yaxis.label.set_color('#606060')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    ax.grid(color='#e7e7e7', linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title(ax.get_title(), loc='left', color='#2d9563', weight='bold', fontsize=11)
    ax.set_title('', loc='center')


def fit_loss_axis(ax, rows):
    """Muestra toda la BCE, que puede superar uno, con margen superior."""
    maximum = max(r[f'{split}_loss'] for r in rows for split in ('train', 'val'))
    ax.set_ylim(0, max(1, maximum * 1.05))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def save_checkpoint(folder, state, improved):
    import torch
    buffer = io.BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()
    # El numerado es el punto de recuperación si se interrumpe la actualización de aliases.
    atomic(folder / 'checkpoints' / f'epoch_{state["epoch"]:03d}.pt', data)
    atomic(folder / 'checkpoints' / 'last.pt', data)
    if improved:
        atomic(folder / 'checkpoints' / 'best.pt', data)


def save_history(folder, history):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    atomic(folder / 'metrics.csv', stream.getvalue().encode('utf-8'))


def confusion_report(path, counts, title, threshold):
    """Matriz y métricas derivadas de recuentos por paciente, sin predicciones."""
    prepare_plotting()
    import matplotlib.pyplot as plt
    tn, fp, fn, tp = (int(counts[k]) for k in ('tn', 'fp', 'fn', 'tp'))
    def ratio(a, b):
        return a / b if b else 0.0
    rows = []
    for label, correct, false_positive, false_negative in (
            ('no pCR (0)', tn, fn, fp), ('pCR (1)', tp, fp, fn)):
        rows.append([label, ratio(correct, correct + false_positive),
                     ratio(correct, correct + false_negative),
                     ratio(2 * correct, 2 * correct + false_positive + false_negative),
                     correct + false_negative])
    accuracy = ratio(tn + tp, tn + fp + fn + tp)
    macro_f1 = (rows[0][3] + rows[1][3]) / 2
    balanced = (rows[0][2] + rows[1][2]) / 2
    fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained',
                                     gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle(title, fontsize=12, weight='bold', color='#2d9563')
    matrix = [[tn, fp], [fn, tp]]
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(tn, fp, fn, tp, 1))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=22,
                    color='white' if matrix[i][j] > max(tn, fp, fn, tp) / 2 else '#253746')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['no pCR', 'pCR'],
           yticklabels=['no pCR', 'pCR'], xlabel='Predicción', ylabel='Clase real')
    table_ax.axis('off')
    text = [[r[0], *[f'{v:.3f}'.replace('.', ',') for v in r[1:4]], str(r[4])] for r in rows]
    table = table_ax.table(cellText=text, colLabels=['Clase', 'Precision', 'Recall', 'F1', 'Pacientes'],
                          cellLoc='center', loc='center', bbox=[0, .42, 1, .3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row == 0:
            cell.set_facecolor('#e8f2ed')
            cell.set_text_props(weight='bold')
    table_ax.text(0, .3, f'Accuracy: {accuracy:.3f}    Macro-F1: {macro_f1:.3f}\n'
                  f'Balanced accuracy: {balanced:.3f}\n'
                  f'Sensibilidad pCR: {rows[1][2]:.3f}    Especificidad: {rows[0][2]:.3f}\n'
                  f'Umbral: {threshold:g} · media de probabilidades por paciente\n'
                  'Métricas sin denominador definido se muestran como 0.', fontsize=10,
                  linespacing=1.7, va='top', transform=table_ax.transAxes)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=140, facecolor='white')
    plt.close(fig)
    atomic(path, buffer.getvalue())


def plot_epoch(folder, history, roc, best_epoch):
    prepare_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    blue, orange = '#2c7ddd', '#f46632'
    epoch = history[-1]['epoch']
    config = json.loads((folder.parent / 'config.json').read_text(encoding='utf-8'))
    x = [r['epoch'] for r in history]
    best = next(r for r in history if r['epoch'] == best_epoch)
    destination = folder / 'figures' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)

    def style(ax, epochs=True):
        style_axes(ax)
        if epochs:
            ax.set_xlabel('Época')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.axvline(best_epoch, color='#aaaaaa', linestyle=':', linewidth=.8)

    def save(fig, name):
        for path in (destination / name, folder / 'figures' / name):
            stream = io.BytesIO()
            fig.savefig(stream, format='png', dpi=140, facecolor='white')
            atomic(path, stream.getvalue())
        plt.close(fig)

    for metric, title in [('loss', 'Pérdida BCE por corte (modo evaluación)'),
                          ('precision', 'Precisión de pCR por paciente'),
                          ('accuracy', 'Accuracy por paciente'), ('auc', 'ROC-AUC por paciente')]:
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        fig.suptitle(config['name'], fontsize=11)
        for split, label, color in [('train', 'entrenamiento', blue), ('val', 'validación (fold de train)', orange)]:
            ax.plot(x, [r[f'{split}_{metric}'] for r in history], color=color, label=label)
        ax.scatter([best_epoch], [best[f'val_{metric}']], color=orange, edgecolor='white', s=45, zorder=5, label='época elegida')
        suffix = f' · umbral {config["evaluation"]["threshold"]:g}' if metric in ('precision', 'accuracy') else ''
        ax.set(title=f'{folder.name} · {title}{suffix}', ylabel=metric, ylim=(0, 1) if metric in ('loss', 'auc') else (0, 1.05))
        style(ax)
        if metric == 'loss':
            fit_loss_axis(ax, history)
        ax.legend()
        save(fig, f'{metric}.png')
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    fig.suptitle(config['name'], fontsize=11)
    for split, label, color in [('train', 'entrenamiento', blue), ('val', 'validación (fold de train)', orange)]:
        ax.plot(roc[split]['fpr'], roc[split]['tpr'], color=color,
                label=f'{label} · AUC {history[-1][f"{split}_auc"]:.3f}')
    ax.plot([0, 1], [0, 1], ':', color='gray', label='azar')
    ax.set(title=f'{folder.name} · ROC por paciente · época {epoch}', xlabel='1 − especificidad',
           ylabel='Sensibilidad pCR', xlim=(0, 1), ylim=(0, 1.02))
    style(ax, False)
    ax.legend()
    save(fig, 'roc.png')
    for row, name in ((history[-1], 'confusion_matrix.png'), (best, 'confusion_matrix_best.png')):
        counts = {k: row[f'val_{k}'] for k in ('tn', 'fp', 'fn', 'tp')}
        title = f'{config["name"]} · {folder.name} · validación · época {int(row["epoch"])}'
        confusion_report(folder / 'figures' / name, counts, title, config['evaluation']['threshold'])
    atomic(destination / 'confusion_matrix.png', (folder / 'figures/confusion_matrix.png').read_bytes())


def plot_summary(run):
    import statistics
    prepare_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    histories = []
    for path in sorted(run.glob('fold_*/metrics.csv')):
        with path.open(encoding='utf-8') as stream:
            histories.append([{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)])
    if not histories:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout='constrained')
    chosen = []
    for i, rows in enumerate(histories):
        best = max(rows, key=lambda r: r['val_auc'])
        chosen.append(best)
        for ax, metric in zip(axes[:2], ['loss', 'auc']):
            for split, label, color in [('train', 'entrenamiento', '#609deb'), ('val', 'validación (fold de train)', '#ff693b')]:
                ax.plot([r['epoch'] for r in rows], [r[f'{split}_{metric}'] for r in rows],
                        color=color, linewidth=1.5, label=label if i == 0 else None)
            ax.scatter([best['epoch']], [best[f'val_{metric}']], color='#f46632', edgecolor='white', s=45, zorder=5,
                       label='época elegida' if i == 0 else None)
    for i, (split, label, color) in enumerate([('train', 'entrenamiento', '#2c7ddd'), ('val', 'validación', '#f46632')]):
        values = [b[f'{split}_auc'] for b in chosen]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values)>1 else 0
        axes[2].bar(i, mean, yerr=std, capsize=5, width=.55, color=color)
        axes[2].scatter([i]*len(values), values, s=15, color='black', zorder=5)
        axes[2].text(i, mean+std+.025, (f'{mean:.2f} ± {std:.2f}' if len(values)>1 else f'{mean:.2f} (n=1)').replace('.', ','), ha='center', weight='bold', fontsize=10)
    axes[0].set(title='Folds · pérdida de entrenamiento y validación', ylabel='Pérdida BCE por corte', ylim=(0, 1))
    fit_loss_axis(axes[0], [r for rows in histories for r in rows])
    axes[1].set(title='Folds · AUC de entrenamiento y validación', ylabel='ROC-AUC por paciente', ylim=(0, 1))
    axes[2].set(title='AUC en la época elegida · media ± DE', ylabel='ROC-AUC en la época elegida', ylim=(0, 1.15),
                xticks=[0, 1], xticklabels=['entrenamiento', 'validación'], xlabel='Test reservado')
    for i, ax in enumerate(axes):
        style_axes(ax)
        if i < 2:
            ax.set_xlabel('Época')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=9, frameon=False, loc='upper right')
        if i > 0:
            ax.axhline(.5, color='gray', linestyle=':')
    gap = statistics.mean(b['train_auc'] - b['val_auc'] for b in chosen)
    axes[2].text(.5, .91, f'entrenamiento − validación = {gap:.2f}'.replace('.', ','),
                 transform=axes[2].transAxes, ha='center', color='#2d9563', weight='bold', fontsize=10)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    fig.suptitle(f'{config["name"]} · {len(histories)}/{len(config["folds"])} folds disponibles · desarrollo, pueden estar en curso', weight='bold')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=140, facecolor='white')
    plt.close(fig)
    atomic(run / 'curves_folds.png', buffer.getvalue())
    counts = {k: sum(b[f'val_{k}'] for b in chosen) for k in ('tn', 'fp', 'fn', 'tp')}
    confusion_report(run / 'confusion_matrix.png', counts,
                     f'{config["name"]} · validación agregada · {len(chosen)}/{len(config["folds"])} folds\n'
                     'Mejor época por AUC de cada fold · provisional mientras avanza el entrenamiento',
                     config['evaluation']['threshold'])
