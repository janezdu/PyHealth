"""Figure 1 from the uniform batch-128 runs, with error bars.

Reads what run_fig1_uniform.sh writes -- test1_rwes128_{spec,gen}_b200.json,
arms named ``<arm>_rep<N>`` -- and emits both a figure and a numbers file:

    _outputs/viz/frontier_b128rwes.html
    _outputs/viz/frontier_b128rwes.json

AGGREGATION, which is the easy thing to get wrong: the main table is the MEAN
over the 8 hospitals of ``PrevVal_All_Prevalence_R2``. Median, and the
rare-code columns, give materially different numbers -- for FedLIBRA the four
statistics span 0.510 to 0.735 on the same run, which is wider than most
arm-to-arm gaps. Both panels are plotted, but the All Codes panel is the one
that lines up with Table 1.

Error bars are mean +/- sample SD over the three independent training runs.
They are NOT seeds: SEED is hardcoded to 0 and the variation comes from
training nondeterminism.

Run from the repo root:  python examples/fedpyhealth/scripts/fig_frontier_uniform.py
"""
import json
import pathlib
import statistics as st

T = '_outputs/results/tests/'
A = 'PrevVal_All_Prevalence_R2'
R = 'PrevVal_Rare_Prevalence_R2'
TAG = 'rwes128'
REPS = (1, 2, 3)

ARMS = [('local', 'local-only', '#4b5563', 'ring'),
        ('centralized', 'centralized', '#1e3a8a', 'ring'),
        ('fedavg', 'FedAvg', '#2a78d6', 'o'),
        ('ft_full', 'FedAvg + full FT', '#c2690a', 'o'),
        ('ft_lora', 'FedAvg + LoRA FT', '#7c3aed', 'o'),
        ('fedlibra', 'FedLIBRA', '#15803d', 'star')]


def load(fold):
    with open(f'{T}test1_{TAG}_{fold}_b200.json') as fh:
        return json.load(fh)['runs']


spec, gen = load('spec'), load('gen')


def per_run(runs, arm, key):
    """One number per training run: mean over that run's 8 hospitals."""
    out = []
    for r in REPS:
        name = f'{arm}_rep{r}'
        if name not in runs:
            continue
        sites = runs[name]
        out.append(st.mean(v[key][0] for v in sites.values()))
    return out


data, summary = {}, {}
for arm, label, _, _ in ARMS:
    d = {}
    for panel, key in (('all', A), ('rare', R)):
        gs, ss = per_run(gen, arm, key), per_run(spec, arm, key)
        if not gs or not ss:
            raise SystemExit(f'no runs found for {arm}; did scoring finish?')
        d[panel] = (st.mean(gs), st.mean(ss),
                    st.stdev(gs) if len(gs) > 1 else 0.0,
                    st.stdev(ss) if len(ss) > 1 else 0.0)
    data[arm] = d
    summary[label] = {
        'n_runs': len(per_run(gen, arm, A)),
        'all':  {'generalized': [d['all'][0], d['all'][2]],
                 'personalized': [d['all'][1], d['all'][3]]},
        'rare': {'generalized': [d['rare'][0], d['rare'][2]],
                 'personalized': [d['rare'][1], d['rare'][3]]},
    }

PW, PH, TOP, LEFT, GAP = 400, 330, 54, 82, 120


def bounds(panel):
    xs = [data[a][panel][0] for a, *_ in ARMS]
    ys = [data[a][panel][1] for a, *_ in ARMS]
    ex = [data[a][panel][2] for a, *_ in ARMS]
    ey = [data[a][panel][3] for a, *_ in ARMS]
    x0, x1 = min(x - e for x, e in zip(xs, ex)), max(x + e for x, e in zip(xs, ex))
    y0, y1 = min(y - e for y, e in zip(ys, ey)), max(y + e for y, e in zip(ys, ey))
    px, py = (x1 - x0) * 0.14 + 0.05, (y1 - y0) * 0.14 + 0.05
    return x0 - px, x1 + px, y0 - py, y1 + py


def star(cx, cy, r, col):
    import math
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append(f'{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}')
    return f'<polygon points="{" ".join(pts)}" fill="{col}"/>'


def panel_svg(panel, title, ox):
    x0, x1, y0, y1 = bounds(panel)
    sx = lambda v: ox + (v - x0) / (x1 - x0) * PW
    sy = lambda v: TOP + PH - (v - y0) / (y1 - y0) * PH
    o = [f'<rect x="{ox}" y="{TOP}" width="{PW}" height="{PH}" fill="var(--panel)" '
         f'stroke="var(--line)" stroke-width="1"/>',
         f'<text x="{ox + PW / 2:.0f}" y="{TOP - 22}" text-anchor="middle" '
         f'font-size="17" font-weight="600" fill="var(--ink)">{title}</text>']

    def ticks(lo, hi):
        span = hi - lo
        step = next(s for s in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0) if span / s <= 7)
        t = -10.0
        while t <= hi:
            if lo < t < hi:
                yield t
            t += step
    for t in ticks(x0, x1):
        o.append(f'<line x1="{sx(t):.1f}" y1="{TOP}" x2="{sx(t):.1f}" y2="{TOP + PH}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text x="{sx(t):.1f}" y="{TOP + PH + 18}" text-anchor="middle" '
                 f'font-size="12" fill="var(--muted)">{t:g}</text>')
    for t in ticks(y0, y1):
        o.append(f'<line x1="{ox}" y1="{sy(t):.1f}" x2="{ox + PW}" y2="{sy(t):.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text x="{ox - 9}" y="{sy(t) + 4:.1f}" text-anchor="end" '
                 f'font-size="12" fill="var(--muted)">{t:g}</text>')

    lg, ls = data['local'][panel][0], data['local'][panel][1]
    cg, cs = data['centralized'][panel][0], data['centralized'][panel][1]
    m = (cs - ls) / (cg - lg)
    fy = lambda g: ls + m * (g - lg)
    o.append(f'<line x1="{sx(x0):.1f}" y1="{sy(fy(x0)):.1f}" x2="{sx(x1):.1f}" '
             f'y2="{sy(fy(x1)):.1f}" stroke="var(--frontier)" stroke-width="2" '
             f'stroke-dasharray="8 5"/>')
    o.append(f'<text x="{ox + PW - 10}" y="{TOP + 17}" text-anchor="end" font-size="12" '
             f'fill="var(--muted)">Beyond the frontier</text>')

    for arm, label, col, mk in ARMS:
        gx, sv, egx, esv = data[arm][panel]
        X, Y = sx(gx), sy(sv)
        if egx > 0:
            o.append(f'<line x1="{sx(gx - egx):.1f}" y1="{Y:.1f}" x2="{sx(gx + egx):.1f}" '
                     f'y2="{Y:.1f}" stroke="{col}" stroke-width="1.6" opacity="0.85"/>')
        if esv > 0:
            o.append(f'<line x1="{X:.1f}" y1="{sy(sv - esv):.1f}" x2="{X:.1f}" '
                     f'y2="{sy(sv + esv):.1f}" stroke="{col}" stroke-width="1.6" opacity="0.85"/>')
        if mk == 'ring':
            o.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="var(--panel)" '
                     f'stroke="{col}" stroke-width="2.4"/>')
        elif mk == 'star':
            o.append(star(X, Y, 11, col))
        else:
            o.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="6.5" fill="{col}"/>')
    return '\n'.join(o)


body = [panel_svg('all', 'All Codes', LEFT),
        panel_svg('rare', 'Rare Codes', LEFT + PW + GAP)]

leg, lx, ly = [], LEFT, TOP + PH + 58
for i, (arm, label, col, mk) in enumerate(ARMS):
    if i == 3:
        lx, ly = LEFT, ly + 26
    if mk == 'ring':
        leg.append(f'<circle cx="{lx + 7}" cy="{ly - 4}" r="6.5" fill="var(--panel)" '
                   f'stroke="{col}" stroke-width="2.4"/>')
    elif mk == 'star':
        leg.append(star(lx + 7, ly - 4, 9, col))
    else:
        leg.append(f'<circle cx="{lx + 7}" cy="{ly - 4}" r="6" fill="{col}"/>')
    leg.append(f'<text x="{lx + 21}" y="{ly}" font-size="13" fill="var(--ink)">{label}</text>')
    lx += 34 + len(label) * 7.3

W, H = LEFT + 2 * PW + GAP + 30, TOP + PH + 112
n = summary['FedLIBRA']['n_runs']
svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="Two scatter panels, All Codes and Rare Codes, of personalized against generalized prevalence R-squared for six arms, every arm trained at batch 128 with rare-code upweighting and early stopping. Bars are the spread over {n} independent training runs.">
{chr(10).join(body)}
{chr(10).join(leg)}
<text x="{LEFT + PW + GAP / 2 + PW / 2:.0f}" y="{TOP + PH + 36}" text-anchor="middle"
  font-size="14" font-weight="600" fill="var(--ink)">Generalized fidelity (R² vs. identity)</text>
<text x="18" y="{TOP + PH / 2:.0f}" text-anchor="middle" font-size="14" font-weight="600"
  fill="var(--ink)" transform="rotate(-90 18 {TOP + PH / 2:.0f})">Personalized fidelity (R² vs. identity)</text>
</svg>'''

html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uniform Batch128 Frontier</title>
<style>
  :root {{ --bg:#fbfaf6; --panel:#ffffff; --ink:#182238; --muted:#5a6478;
    --line:#ddd9cd; --grid:#eceadf; --frontier:#8a8578; }}
  :root:not([data-theme="light"]) {{ @media (prefers-color-scheme: dark) {{
    --bg:#14161c; --panel:#1b1e26; --ink:#eef1f6; --muted:#a2abbd;
    --line:#333843; --grid:#272c35; --frontier:#8e95a3; }} }}
  :root[data-theme="dark"] {{ --bg:#14161c; --panel:#1b1e26; --ink:#eef1f6;
    --muted:#a2abbd; --line:#333843; --grid:#272c35; --frontier:#8e95a3; }}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:28px 16px 44px;background:var(--bg);color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1040px;margin:0 auto}}
  h1{{font-size:1.4rem;line-height:1.25;margin:0 0 6px;font-weight:650}}
  .sub{{color:var(--muted);font-size:.95rem;line-height:1.55;margin:0 0 20px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:12px 10px 4px}}
  svg{{display:block;width:100%;height:auto}}
  .note{{margin:20px 0 0;font-size:.92rem;line-height:1.62;color:var(--muted)}}
  .note b{{color:var(--ink);font-weight:600}}
</style>
</head>
<body>
<div class="wrap">
  <h1>All six arms, one matched setting</h1>
  <p class="sub">Batch 128, rare-code upweighting and early stopping on every arm.
  Bars are mean ± SD over {n} independent training runs. The dashed line is the
  frontier through local-only and centralized.</p>
  <div class="card">{svg}</div>
  <p class="note">
    <b>Aggregated as the main table is:</b> mean over the 8 hospitals of all-codes
    R², then mean and spread over the runs. Median, and the rare-code columns,
    give different numbers — for one FedLIBRA run the four statistics span 0.510
    to 0.735, wider than most arm-to-arm gaps. The All Codes panel is the one
    that lines up with Table 1.
    <br><br>
    <b>Independent training runs, not seeds.</b> SEED is hardcoded to 0; the
    variation is training nondeterminism.
  </p>
</div>
</body>
</html>'''

viz = pathlib.Path('_outputs/viz')
viz.mkdir(parents=True, exist_ok=True)
(viz / 'frontier_b128rwes.html').write_text(html)
(viz / 'frontier_b128rwes.json').write_text(json.dumps(summary, indent=2))
print('wrote _outputs/viz/frontier_b128rwes.html')
print('wrote _outputs/viz/frontier_b128rwes.json')
for label, v in summary.items():
    a = v['all']
    print('  %-18s n=%d  spec %+.3f +/- %.3f   gen %+.3f +/- %.3f'
          % (label, v['n_runs'], a['personalized'][0], a['personalized'][1],
             a['generalized'][0], a['generalized'][1]))
