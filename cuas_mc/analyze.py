#!/usr/bin/env python3
"""Aggregate results/, compute Pk with Wilson 95% CIs, plot flail vs ram."""
import json
import pathlib
from collections import defaultdict
from math import sqrt
import matplotlib
matplotlib.use('Agg')   # headless: save PNG without a display
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parent / 'results'


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    data = defaultdict(lambda: {'contact': 0, 'geom': 0, 'n': 0})
    for f in RESULTS.glob('*.json'):
        r = json.loads(f.read_text())
        key = (r['interceptor_type'], round(r['error_sigma'], 3))
        data[key]['n'] += 1
        data[key]['contact'] += bool(r.get('kill_contact'))
        data[key]['geom'] += bool(r.get('kill_geometric'))

    fig, ax = plt.subplots(figsize=(9, 6))
    for itype, color in (('flail', 'tab:red'), ('ram', 'tab:blue')):
        sig = sorted(s for (t, s) in data if t == itype)
        if not sig:
            continue
        for crit, ls in (('contact', '-'), ('geom', '--')):
            pk, lo, hi = zip(*(wilson(data[(itype, s)][crit], data[(itype, s)]['n'])
                               for s in sig))
            ax.plot([s * 100 for s in sig], pk, ls, color=color,
                    marker='o' if crit == 'contact' else 's',
                    label=f'{itype} ({crit})')
            ax.fill_between([s * 100 for s in sig], lo, hi, color=color, alpha=0.12)
        print(f'--- {itype} ---')
        for s in sig:
            d = data[(itype, s)]
            print(f'  sigma={s*100:.0f}cm  n={d["n"]}  '
                  f'Pk_contact={d["contact"]/d["n"]:.3f}  '
                  f'Pk_geom={d["geom"]/d["n"]:.3f}')

    ax.set_xlabel('Guidance error sigma (cm, per axis)')
    ax.set_ylabel('Probability of Kill')
    ax.set_title('Flail vs Ram Pk — contact-based (0.25 ms timestep) vs geometric')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / 'pk_curves.png', dpi=200)
    print(f'\nSaved {RESULTS / "pk_curves.png"}')


if __name__ == '__main__':
    main()