import json, statistics as st, pathlib

T = '_outputs/results/tests/'
A = 'PrevVal_All_Prevalence_R2'
R = 'PrevVal_Rare_Prevalence_R2'


def agg(f, arm, key):
    d = json.load(open(T + f + '.json'))['runs'][arm]
    return st.mean([v[key][0] for v in d.values()])


ARMS = [('fedavg', 'FedAvg', '#1d4ed8'),
        ('ft_full', 'FedAvg + full FT', '#c2690a'),
        ('ft_lora', 'FedAvg + LoRA FT', '#7c3aed'),
        ('fedlibra', 'FedLIBRA', '#15803d')]

data = {}
for arm, _, _ in ARMS:
    for rep in (1, 2, 3):
        n = f'{arm}_rep{rep}'
        data[n] = {
            'all': (agg('test1_repsfed4_gen_b200', n, A), agg('test1_repsfed4_spec_b200', n, A)),
            'rare': (agg('test1_repsfed4_gen_b200', n, R), agg('test1_repsfed4_spec_b200', n, R)),
        }
for n, src in [('local', 'loc_unw_full'), ('centralized', 'cent_rw_es')]:
    data[n] = {
        'all': (agg('test1_rwes2x2_gen_b200', src, A), agg('test1_rwes2x2_spec_b200', src, A)),
        'rare': (agg('test1_rwes2x2_gen_b200', src, R), agg('test1_rwes2x2_spec_b200', src, R)),
    }

PANELS = [('all', 'All Codes', -1.8, 1.05, -0.72, 0.92),
          ('rare', 'Rare Codes', -2.9, 1.05, -1.65, 0.72)]

PW, PH = 400, 330           # plot area
TOP, LEFT, GAP = 54, 76, 118


def panel_svg(key, title, x0, x1, y0, y1, ox):
    sx = lambda v: ox + (v - x0) / (x1 - x0) * PW
    sy = lambda v: TOP + PH - (v - y0) / (y1 - y0) * PH
    o = []
    o.append(f'<rect x="{ox}" y="{TOP}" width="{PW}" height="{PH}" fill="var(--panel)" '
             f'stroke="var(--line)" stroke-width="1"/>')
    o.append(f'<text x="{ox + PW/2:.0f}" y="{TOP-22}" text-anchor="middle" font-size="17" '
             f'font-weight="600" fill="var(--ink)">{title}</text>')

    # gridlines + ticks
    step = 1.0 if key == 'all' else 1.0
    t = -3.0
    while t <= x1:
        if x0 < t < x1:
            o.append(f'<line x1="{sx(t):.1f}" y1="{TOP}" x2="{sx(t):.1f}" y2="{TOP+PH}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
            o.append(f'<text x="{sx(t):.1f}" y="{TOP+PH+18}" text-anchor="middle" font-size="12" '
                     f'fill="var(--muted)">{t:.0f}</text>')
        t += step
    t = -2.0
    while t <= y1:
        if y0 < t < y1:
            o.append(f'<line x1="{ox}" y1="{sy(t):.1f}" x2="{ox+PW}" y2="{sy(t):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
            o.append(f'<text x="{ox-9}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12" '
                     f'fill="var(--muted)">{t:.0f}</text>')
        t += 0.5

    # frontier line through the two anchors, extended across the panel
    lg, ls = data['local'][key]
    cg, cs = data['centralized'][key]
    m = (cs - ls) / (cg - lg)
    fy = lambda gx: ls + m * (gx - lg)
    o.append(f'<line x1="{sx(x0):.1f}" y1="{sy(fy(x0)):.1f}" x2="{sx(x1):.1f}" y2="{sy(fy(x1)):.1f}" '
             f'stroke="var(--frontier)" stroke-width="2" stroke-dasharray="8 5"/>')
    o.append(f'<text x="{ox+PW-10}" y="{TOP+16}" text-anchor="end" font-size="12" '
             f'fill="var(--muted)">Beyond the frontier</text>')

    # anchors
    for nm, lbl in [('local', 'local-only'), ('centralized', 'centralized')]:
        gx, sv = data[nm][key]
        o.append(f'<circle cx="{sx(gx):.1f}" cy="{sy(sv):.1f}" r="6" fill="none" '
                 f'stroke="var(--anchor)" stroke-width="2"/>')
        o.append(f'<text x="{sx(gx):.1f}" y="{sy(sv)-12:.1f}" text-anchor="middle" font-size="11.5" '
                 f'fill="var(--muted)">{lbl}</text>')

    # arms: rep1 open, reps 2-3 filled, dashed connector showing the drift
    for arm, lbl, col in ARMS:
        p = [data[f'{arm}_rep{r}'][key] for r in (1, 2, 3)]
        mx = (p[1][0] + p[2][0]) / 2
        my = (p[1][1] + p[2][1]) / 2
        o.append(f'<line x1="{sx(p[0][0]):.1f}" y1="{sy(p[0][1]):.1f}" x2="{sx(mx):.1f}" '
                 f'y2="{sy(my):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="3 3" '
                 f'opacity="0.75"/>')
        o.append(f'<circle cx="{sx(p[0][0]):.1f}" cy="{sy(p[0][1]):.1f}" r="6.5" fill="var(--panel)" '
                 f'stroke="{col}" stroke-width="2.2"/>')
        for gx, sv in p[1:]:
            o.append(f'<circle cx="{sx(gx):.1f}" cy="{sy(sv):.1f}" r="5.5" fill="{col}"/>')
    return '\n'.join(o)


body = [panel_svg(k, t, x0, x1, y0, y1, LEFT + i * (PW + GAP))
        for i, (k, t, x0, x1, y0, y1) in enumerate(PANELS)]

legend = []
lx = LEFT
for arm, lbl, col in ARMS:
    legend.append(f'<circle cx="{lx+7}" cy="{TOP+PH+56}" r="5.5" fill="{col}"/>')
    legend.append(f'<text x="{lx+20}" y="{TOP+PH+60}" font-size="13" fill="var(--ink)">{lbl}</text>')
    lx += 25 + len(lbl) * 7.4
legend.append(f'<circle cx="{LEFT+7}" cy="{TOP+PH+84}" r="6.5" fill="var(--panel)" '
              f'stroke="var(--muted)" stroke-width="2.2"/>')
legend.append(f'<text x="{LEFT+20}" y="{TOP+PH+88}" font-size="13" fill="var(--ink)">'
              f'rep 1 — the published run (Aug 27–Sep 7 code)</text>')
legend.append(f'<circle cx="{LEFT+372}" cy="{TOP+PH+84}" r="5.5" fill="var(--muted)"/>')
legend.append(f'<text x="{LEFT+385}" y="{TOP+PH+88}" font-size="13" fill="var(--ink)">'
              f'reps 2 and 3 — retrained on current code</text>')

W = LEFT + 2 * PW + GAP + 30
H = TOP + PH + 110

svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="Two scatter panels, All Codes and Rare Codes, plotting personalized against generalized prevalence R-squared. Each of the four federated arms appears three times: the published run as an open circle and two retrained runs as filled circles, joined by a dashed line. For FedAvg, FedAvg plus full fine-tune and FedAvg plus LoRA fine-tune the published run sits far from the two retrained runs, which coincide. FedLIBRA's three runs coincide.">
{chr(10).join(body)}
{chr(10).join(legend)}
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
<title>Replicate Runs Scatter</title>
<style>
  :root {{
    --bg:#fbfaf6; --panel:#ffffff; --ink:#182238; --muted:#5a6478;
    --line:#ddd9cd; --grid:#eceadf; --frontier:#8a8578; --anchor:#6b7280;
  }}
  :root:not([data-theme="light"]) {{
    @media (prefers-color-scheme: dark) {{
      --bg:#14161c; --panel:#1b1e26; --ink:#eef1f6; --muted:#a2abbd;
      --line:#333843; --grid:#272c35; --frontier:#8e95a3; --anchor:#9aa3b2;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#14161c; --panel:#1b1e26; --ink:#eef1f6; --muted:#a2abbd;
    --line:#333843; --grid:#272c35; --frontier:#8e95a3; --anchor:#9aa3b2;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:28px 16px 44px;background:var(--bg);color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1040px;margin:0 auto}}
  h1{{font-size:1.4rem;line-height:1.25;margin:0 0 6px;font-weight:650}}
  .sub{{color:var(--muted);font-size:.95rem;line-height:1.55;margin:0 0 20px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 10px 4px}}
  svg{{display:block;width:100%;height:auto}}
  .note{{margin:20px 0 0;font-size:.92rem;line-height:1.62;color:var(--muted)}}
  .note b{{color:var(--ink);font-weight:600}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Three independent training runs per arm</h1>
  <p class="sub">Each arm is plotted three times. The dashed grey line is the frontier through
  local-only and centralized. Spread within an arm is what an error bar on Figure 1 would show.</p>
  <div class="card">{svg}</div>
  <p class="note">
    <b>FedLIBRA is the only arm that reproduces.</b> Its three runs sit on top of each other
    (SD 0.002 on both axes). For the other three arms the published run sits far from the two
    retrained runs, which land almost exactly on each other — not the pattern random variation
    produces.
    <br><br>
    <b>That gap is code drift, not noise.</b> Those three runs date from 27–30 August;
    <i>train.py</i> changed on 11 and 13 September and again this week. FedLIBRA's published run
    is from 7 September and reproduces. Read the filled pairs as the run-to-run spread — roughly
    0.003 to 0.02 — and the open-to-filled distance as the difference between the published code
    and the current code.
  </p>
</div>
</body>
</html>'''

out = pathlib.Path('_outputs/viz/replicates.html')
out.write_text(html)
print('wrote', out, len(html), 'bytes')
