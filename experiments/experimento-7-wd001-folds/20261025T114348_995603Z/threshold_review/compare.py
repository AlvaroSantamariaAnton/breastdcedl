"""Tablas OOF desde conteos completos guardados; sin inferencia ni datos test."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'threshold_comparison/20261026T005412_420724Z'
def read(name):
    with (SOURCE/name).open(newline='', encoding='utf-8') as f:
        return [{k: float(v) if k != 'scope' else v for k, v in row.items()} for row in csv.DictReader(f)]

points = read('operating_points.csv')
assert len(points) == 1098
assert (points[0]['fp'], points[0]['tp']) == (0, 0)
assert (points[-1]['fp'], points[-1]['tp']) == (775, 322)
for a, b in zip(points, points[1:]):
    assert a['threshold'] > b['threshold']
    assert b['fp']-a['fp'] >= 0 and b['tp']-a['tp'] >= 0
    assert b['fp']+b['tp']-a['fp']-a['tp'] == 1  # Ningún punto omitido.

def metrics(threshold):
    row = [r for r in points if r['threshold'] >= threshold][-1]
    tn, fp, fn, tp = (int(row[k]) for k in ('tn','fp','fn','tp'))
    assert tn+fp == 775 and fn+tp == 322
    recall, specificity = tp/322, tn/775
    precision = tp/(tp+fp) if tp+fp else 0
    f1 = 2*tp/(2*tp+fp+fn)
    f1_negative = 2*tn/(2*tn+fp+fn)
    return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                sensitivity=recall, specificity=specificity, precision_pcr=precision,
                f1_pcr=f1, f1_no_pcr=f1_negative, macro_f1=(f1+f1_negative)/2,
                balanced_accuracy=(recall+specificity)/2, accuracy=(tp+tn)/1097)

for old in read('refined_thresholds.csv') + [r for r in read('metrics.csv') if r['scope']=='OOF']:
    new = metrics(old['threshold'])
    assert all(new[k] == old[k] for k in ('tn','fp','fn','tp'))

rows = [metrics(i/100) for i in range(20, 61)]
with (OUT/'thresholds_oof.csv').open('w', newline='', encoding='utf-8') as f:
    writer=csv.DictWriter(f, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
audit = dict(source='operating_points.csv', source_sha256=hashlib.sha256((SOURCE/'operating_points.csv').read_bytes()).hexdigest(),
             scope='pooled OOF, one held-out model per patient; NOT full ensemble', patients=1097,
             complete_points_verified=True, previous_counts_reproduced=True, inference_repeated=False, test_used=False,
             grid='0.20 through 0.60 inclusive, step 0.01', selected_threshold=None,
             best_balanced_accuracy=max(rows,key=lambda r:r['balanced_accuracy']),
             best_macro_f1=max(rows,key=lambda r:r['macro_f1']))
(OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
for r in rows:
    if r['threshold'] in [.20,.25,.30,.32,.33,.34,.35,.36,.37,.38,.40,.45,.50,.60]:
        print(f"{r['threshold']:.2f}: FP={r['fp']}, FN={r['fn']}, sensitivity={r['sensitivity']:.4f}, specificity={r['specificity']:.4f}, macroF1={r['macro_f1']:.4f}, BA={r['balanced_accuracy']:.4f}")
