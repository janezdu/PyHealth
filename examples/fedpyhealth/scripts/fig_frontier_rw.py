import json, statistics as st, pathlib

T = '_outputs/results/tests/'
A = 'PrevVal_All_Prevalence_R2'
R = 'PrevVal_Rare_Prevalence_R2'


def agg(f, arm, key):
    d = json.load(open(T + f + '.json'))['runs'][arm]
    return st.mean([v[key][0] for v in d.values()])


# marker: 'o' filled, 'ring' open circle, 'star'
ARMS = [('rw_local', 'local-only', '#4b5563', 'ring'),
        ('rw_centralized', 'centralized', '#1e3a8a', 'ring'),
        ('rw_fedavg', 'FedAvg', '#2a78d6', 'o'),
        ('rw_ft_full', 'FedAvg + full FT', '#c2690a', 'o'),
        ('rw_ft_lora', 'FedAvg + LoRA FT', '#7c3aed', 'o'),
        ('rw_fedlibra', 'FedLIBRA', '#15803d', 'star')]

data = {}
for a, _, _, _ in ARMS:
    data[a] = {
        'all': (agg('test1_rwfig1_gen_b200', a, A), agg('test1_rwfig1_spec_b200', a, A)),
        'rare': (agg('test1_rwfig1_gen_b200', a, R), agg('test1_rwfig1_spec_b200', a, R)),
    }

PANELS = [('all', 'All Codes', -1.3, 1.1, -0.1, 0.80),
          ('rare', 'Rare Codes', -2.3, 1.1, -0.65, 0.62)]
PW, PH = 400, 330
TOP, LEFT, GAP = 54, 78, 118


def star(cx, cy, r, col):
    import math
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append(f'{cx + rr*math.cos(ang):.1f},{cy + rr*math.sin(ang):.1f}')
    return f'<polygon points="{" ".join(pts)}" fill="{col}"/>'


def panel(key, title, x0, x1, y0, y1, ox):
    sx = lambda v: ox + (v - x0) / (x1 - x0) * PW
    sy = lambda v: TOP + PH - (v - y0) / (y1 - y0) * PH
    o = [f'<rect x="{ox}" y="{TOP}" width="{PW}" height="{PH}" fill="var(--panel)" '
         f'stroke="var(--line)" stroke-width="1"/>',
         f'<text x="{ox+PW/2:.0f}" y="{TOP-22}" text-anchor="middle" font-size="17" '
         f'font-weight="600" fill="var(--ink)">{title}</text>']
    t = -3.0
    while t <= x1:
        if x0 < t < x1:
            o.append(f'<line x1="{sx(t):.1f}" y1="{TOP}" x2="{sx(t):.1f}" y2="{TOP+PH}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
            o.append(f'<text x="{sx(t):.1f}" y="{TOP+PH+18}" text-anchor="middle" font-size="12" '
                     f'fill="var(--muted)">{t:.0f}</text>')
        t += 0.5 if key == 'all' else 1.0
    t = -2.0
    while t <= y1:
        if y0 < t < y1:
            o.append(f'<line x1="{ox}" y1="{sy(t):.1f}" x2="{ox+PW}" y2="{sy(t):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
            o.append(f'<text x="{ox-9}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12" '
                     f'fill="var(--muted)">{t:.1f}</text>')
        t += 0.25 if key == 'all' else 0.25

    lg, ls = data['rw_local'][key]
    cg, cs = data['rw_centralized'][key]
    m = (cs - ls) / (cg - lg)
    fy = lambda g: ls + m * (g - lg)
    o.append(f'<line x1="{sx(x0):.1f}" y1="{sy(fy(x0)):.1f}" x2="{sx(x1):.1f}" '
             f'y2="{sy(fy(x1)):.1f}" stroke="var(--frontier)" stroke-width="2" stroke-dasharray="8 5"/>')
    o.append(f'<text x="{ox+PW-10}" y="{TOP+17}" text-anchor="end" font-size="12" '
             f'fill="var(--muted)">Beyond the frontier</text>')

    for a, lbl, col, mk in ARMS:
        gx, sv = data[a][key]
        X, Y = sx(gx), sy(sv)
        if mk == 'ring':
            o.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="var(--panel)" '
                     f'stroke="{col}" stroke-width="2.4"/>')
        elif mk == 'star':
            o.append(star(X, Y, 11, col))
        else:
            o.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="6.5" fill="{col}"/>')
    return '\n'.join(o)


body = [panel(k, t, a, b, c, d, LEFT + i * (PW + GAP))
        for i, (k, t, a, b, c, d) in enumerate(PANELS)]

leg, lx, ly = [], LEFT, TOP + PH + 58
for i, (a, lbl, col, mk) in enumerate(ARMS):
    if i == 3:
        lx, ly = LEFT, ly + 26
    if mk == 'ring':
        leg.append(f'<circle cx="{lx+7}" cy="{ly-4}" r="6.5" fill="var(--panel)" stroke="{col}" stroke-width="2.4"/>')
    elif mk == 'star':
        leg.append(star(lx + 7, ly - 4, 9, col))
    else:
        leg.append(f'<circle cx="{lx+7}" cy="{ly-4}" r="6" fill="{col}"/>')
    leg.append(f'<text x="{lx+21}" y="{ly}" font-size="13" fill="var(--ink)">{lbl}</text>')
    lx += 34 + len(lbl) * 7.3

W = LEFT + 2 * PW + GAP + 30
H = TOP + PH + 112
svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="Two scatter panels, All Codes and Rare Codes, of personalized against generalized prevalence R-squared for six arms, all trained with rare-code upweighting and early stopping. FedLIBRA has the highest personalized score in both panels and sits above the frontier line. FedAvg has the highest generalized score, above centralized on both axes.">
{chr(10).join(body)}
{chr(10).join(leg)}
<text x="{LEFT + PW + GAP/2 + PW/2:.0f}" y="{TOP+PH+36}" text-anchor="middle" font-size="14"
  font-weight="600" fill="var(--ink)">Generalized fidelity (R² vs. identity)</text>
<text x="18" y="{TOP+PH/2:.0f}" text-anchor="middle" font-size="14" font-weight="600"
  fill="var(--ink)" transform="rotate(-90 18 {TOP+PH/2:.0f})">Personalized fidelity (R² vs. identity)</text>
</svg>'''

html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matched Upweighting Frontier</title>
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
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 10px 4px}}
  .note{{margin:20px 0 0;font-size:.92rem;line-height:1.62;color:var(--muted)}}
  .note b{{color:var(--ink);font-weight:600}}
  svg{{display:block;width:100%;height:auto}}
</style>
</head>
<body>
<div class="wrap">
  <h1>All six arms under matched training</h1>
  <p class="sub">Every arm trained with rare-code upweighting and early stopping, at batch 256 —
  so no arm has a treatment the others lack. The frontier runs through local-only and centralized.</p>
  <div class="card">{svg}</div>
  <p class="note">
    <b>FedLIBRA keeps the best personalized fidelity</b> in both panels (+0.642 all, +0.417 rare)
    and still sits well above the frontier line.
    <br><br>
    <b>But FedAvg now beats centralized on both axes</b> (+0.290 / +0.864 against +0.134 / +0.858).
    Centralized is no longer the generalization ceiling, so it is a weaker right-hand anchor than
    the published figure assumes, and FedLIBRA is no longer the only arm outside the line.
    The defensible claim here is best balance, not uniquely beyond.
    <br><br>
    <b>Single run per arm.</b> These six carry no error bars; the replicate runs were done on the
    unweighted arms at batch 256, not on these.
  </p>
</div>
</body>
</html>'''

out = pathlib.Path('_outputs/viz/frontier_rw.html')
out.write_text(html)
print('wrote', out, len(html), 'bytes')
