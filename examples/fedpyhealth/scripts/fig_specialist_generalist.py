"""Paper figure: is a generator a specialist or a generalist?

Both axes are the *same* prevalence-fidelity metric. Only the real fold each
arm is scored against differs:

``y`` (specialist)  each site's synthetic against its OWN test fold
                    (``test1_prevalence.py --real-scope hospital``)
``x`` (generalist)  every site's synthetic against the SAME pooled cohort test
                    fold (``--real-scope pooled``)

So the identity diagonal is where a generator matches its own site exactly as
well as it matches the federation. Above it the arm is a specialist, below it a
generalist. The point of the figure is that the four standard regimes do not
merely differ in quality -- they *reverse order* between the two targets, which
is the one-dimensional generalization/personalization frontier made visible on
a density-modelling task rather than a discriminative one.

Nothing here recomputes a metric. It reads what ``test1_prevalence.py`` wrote
for the two ``--real-scope`` settings and reduces each arm to the MEDIAN over
hospitals (the mean is dragged somewhere no hospital is by a single wrecked
R^2 -- see ``utils/eda/fidelity.py``).

Hospital identifiers are NOT carried into the output. Sites are relabelled
``H1..H8`` by descending training size, so a committed figure in the paper repo
holds no site identifier -- see ``.llms/rules/03-data-safety.md``.

Usage
-----
::

    .venv/bin/python examples/fedpyhealth/scripts/fig_specialist_generalist.py

CPU-only and about a second: it reads two small JSON files and draws. Writes
``specialist_generalist_rw.png`` and ``.html`` into ``--out-dir``.
"""

import argparse
import json
import os
import statistics as st

#: The four regimes, ordered least-shared to most-shared. The colour ramp
#: encodes that axis, so the figure reads as a ramp in greyscale too.
REGIMES = [
    ("local", "local", "#86b6ef", "8 isolated generators"),
    ("fedavg_ft", "fedavg+ft", "#5598e7", "global model, fine-tuned per site"),
    ("fedavg", "fedavg", "#2a78d6", "one federated generator"),
    ("centralized", "centralized", "#184f95", "one generator, pooled raw data"),
]

#: ``(payload key, test1 metric, axis label, higher-is-better)``. Pearson is the
#: figure's default because R^2 is measured against the identity line and goes
#: far enough negative to need its own scale, which would cost the shared-axis
#: reading the diagonal depends on.
METRICS = [
    ("pearson_rare", "PrevVal_Rare_Prevalence_Pearson",
     "Prevalence Pearson $r$, rare codes", True),
    ("pearson_all", "PrevVal_All_Prevalence_Pearson",
     "Prevalence Pearson $r$, all codes", True),
    ("r2_rare", "PrevVal_Rare_Prevalence_R2", "Prevalence $R^2$, rare codes", True),
    ("r2_all", "PrevVal_All_Prevalence_R2", "Prevalence $R^2$, all codes", True),
    ("rmse_rare", "PrevVal_Rare_Prevalence_RMSE", "Prevalence RMSE, rare codes", False),
    ("rmse_all", "PrevVal_All_Prevalence_RMSE", "Prevalence RMSE, all codes", False),
]

#: Which two metrics the static PNG panels show. The interactive HTML carries
#: all six; a paper figure has to pick, and the head/tail pair is the pick that
#: shows the reversal sharpening on the tail.
PNG_PANELS = ("pearson_all", "pearson_rare")

DEFAULTS = {
    "spec": "_outputs/results/tests/test1_RW_n8000_capped_pooledrare.json",
    "gen": "_outputs/results/tests/test1_RW_pooledreal_pooledrare.json",
    "sizes": "_outputs/results/runs/local_full_hilo8_random_rw.json",
    "out_dir": "paper/figs",
    "stem": "specialist_generalist_rw",
}


def load(spec_path: str, gen_path: str) -> tuple:
    """Read both targets and refuse anything that is not a matched pair.

    The two files differ in exactly one control (``real_scope``). If they also
    differ in cohort, fold, rare scope or synthetic cap then the vertical and
    horizontal positions were measured under different conditions and the
    diagonal means nothing -- so that is an error, not a warning. This project
    has already shipped one result built from mismatched scoring runs.
    """
    for p in (spec_path, gen_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nRun the scoring pass first:\n"
                "  COHORT=hilo8_random SUFFIX=_rw sbatch "
                "examples/fedpyhealth/scripts/score_cohort.sh")
    spec, gen = json.load(open(spec_path)), json.load(open(gen_path))

    if spec.get("real_scope") != "hospital" or gen.get("real_scope") != "pooled":
        raise SystemExit(
            "real_scope mismatch: expected the specialist file to hold "
            f"'hospital' and the generalist file 'pooled', got "
            f"{spec.get('real_scope')!r} and {gen.get('real_scope')!r}. "
            "The two arguments are probably the wrong way round.")
    for key in ("cohort_name", "fold", "rare_scope", "synth_cap"):
        if spec.get(key) != gen.get(key):
            raise SystemExit(
                f"{key} differs between the two targets "
                f"({spec.get(key)!r} vs {gen.get(key)!r}). Both axes must come "
                "from one scoring pass or the diagonal is not a comparison.")
    if set(spec.get("runs", {})) != set(gen.get("runs", {})):
        raise SystemExit("the two targets scored different arms.")
    return spec, gen


def _site_order(spec: dict, sizes_path: str) -> dict:
    """``{hospital_id: "H1"..}`` by descending training size.

    Falls back to the order test1 wrote if the run file is absent; test1
    iterates the cohort manifest, which is already size-ordered, so the labels
    come out the same either way. The run file is preferred because it makes
    the ordering checkable rather than assumed.
    """
    hosps = list(spec["runs"][next(iter(spec["runs"]))].keys())
    if os.path.exists(sizes_path):
        part = json.load(open(sizes_path)).get("partition", {})
        known = [h for h in hosps if h in part]
        if len(known) == len(hosps):
            hosps = sorted(hosps, key=lambda h: -part[h].get("n_train", 0))
    return {h: f"H{i + 1}" for i, h in enumerate(hosps)}


def build(spec: dict, gen: dict, sites: dict) -> dict:
    """Arm medians and per-site points, for every metric, both targets."""
    arms = [r for r in REGIMES if r[0] in spec["runs"]]
    metrics, per_site = {}, {}
    for key, src, label, up in METRICS:
        rows = []
        for arm_id, arm_label, colour, note in arms:
            sv = [v[src][0] for v in spec["runs"][arm_id].values() if src in v]
            gv = [v[src][0] for v in gen["runs"][arm_id].values() if src in v]
            if not sv or not gv:
                continue
            rows.append({"id": arm_id, "label": arm_label, "colour": colour,
                         "note": note, "own": st.median(sv),
                         "pooled": st.median(gv)})
        metrics[key] = {"label": label, "better_up": up, "arms": rows}

        pts = []
        for hid, tag in sites.items():
            for arm_id, arm_label, colour, _n in arms:
                s = spec["runs"][arm_id].get(hid, {}).get(src)
                g = gen["runs"][arm_id].get(hid, {}).get(src)
                if s is None or g is None:
                    continue
                pts.append({"site": tag, "id": arm_id, "label": arm_label,
                            "colour": colour, "own": s[0], "pooled": g[0]})
        per_site[key] = pts

    return {
        "metrics": metrics, "per_site": per_site,
        "metric_order": [m[0] for m in METRICS if m[0] in metrics],
        "meta": {"cohort": spec.get("cohort_name"), "fold": spec.get("fold"),
                 "rare_scope": spec.get("rare_scope"),
                 "synth_cap": spec.get("synth_cap"),
                 "n_sites": len(sites), "n_arms": len(arms),
                 "condition": "rare-code upweighting"},
    }


# --------------------------------------------------------------------------- #
# Static figure                                                                #
# --------------------------------------------------------------------------- #
def write_png(payload: dict, path: str, dpi: int = 300) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.linewidth": 0.7, "axes.edgecolor": "#55534d",
        "xtick.color": "#55534d", "ytick.color": "#55534d",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.labelcolor": "#2b2a27", "text.color": "#2b2a27",
    })
    panels = [k for k in PNG_PANELS if k in payload["metrics"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 3.55))
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, key in zip(axes, panels):
        m = payload["metrics"][key]
        pts = payload["per_site"][key]
        vals = ([a["own"] for a in m["arms"]] + [a["pooled"] for a in m["arms"]]
                + [p["own"] for p in pts] + [p["pooled"] for p in pts])
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.09 or abs(hi) * 0.09 or 0.1
        lo, hi = lo - pad, hi + pad

        # The diagonal first, so every marker sits on top of it.
        ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.9, color="#a9a79d",
                zorder=1)
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="#8fb6de",
                        alpha=0.07, lw=0, zorder=0)

        for p in pts:
            ax.plot(p["pooled"], p["own"], "o", ms=3.1, color=p["colour"],
                    alpha=0.45, mec="white", mew=0.45, zorder=2)
        for a in m["arms"]:
            ax.plot(a["pooled"], a["own"], "o", ms=8.5, color=a["colour"],
                    mec="white", mew=1.3, zorder=4)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("scored against the pooled cohort fold\n(generalist)")
        ax.set_ylabel("scored against the site's own fold\n(specialist)")
        ax.set_title(m["label"].replace("$r$", "r"), fontsize=8.5, pad=7,
                     color="#2b2a27")
        ax.grid(True, color="#e6e4dc", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # A white halo, because at these axis limits a corner label lands on
        # top of a site dot in at least one panel.
        halo = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.8)
        ax.annotate("fits its own site better", xy=(0.04, 0.96),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.8, color="#7c7a72", style="italic",
                    bbox=halo, zorder=5)
        ax.annotate("fits the cohort better", xy=(0.96, 0.035),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=6.8, color="#7c7a72", style="italic",
                    bbox=halo, zorder=5)

    handles = [Line2D([], [], marker="o", ls="", ms=6.5, mec="white", mew=1.0,
                      color=a["colour"], label=a["label"])
               for a in payload["metrics"][panels[0]]["arms"]]
    handles.append(Line2D([], [], marker="o", ls="", ms=3.4, color="#8a8880",
                          alpha=0.55, label="individual site"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=7.6, handletextpad=0.35,
               columnspacing=1.3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interactive figure                                                           #
# --------------------------------------------------------------------------- #
HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Specialist or Generalist &middot; rare-upweighted</title>
<style>
:root{--bg:#faf9f5;--surface:#fff;--ink:#2b2a27;--ink-2:#55534d;--ink-3:#898781;
--line:#e6e4dc;--grid:#eeece4;--axis:#a9a79d;--diag:#c3c2b7;}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1a;--surface:#242422;
--ink:#eeece4;--ink-2:#c3c2b7;--ink-3:#898781;--line:#2c2c2a;--grid:#2c2c2a;
--axis:#7c7a72;--diag:#4a4a46;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:28px 20px 48px}
h1{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.eyebrow{margin:0 0 20px;color:var(--ink-3);font-size:12px;
text-transform:uppercase;letter-spacing:.07em}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:18px}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin-bottom:10px}
label{display:block;font-size:11px;color:var(--ink-3);margin-bottom:3px}
select,button{font:inherit;font-size:13px;padding:5px 9px;border-radius:6px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button[aria-pressed=true]{background:#2a78d6;border-color:#2a78d6;color:#fff}
.plotwrap{position:relative}
svg{width:100%;height:auto;display:block;overflow:visible}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
background:var(--surface);border:1px solid var(--line);border-radius:7px;
padding:7px 9px;font-size:12px;box-shadow:0 5px 18px rgba(0,0,0,.13);
white-space:nowrap;z-index:5}
.tip b{font-family:ui-monospace,Menlo,monospace}
.tip .r{display:flex;gap:12px;justify-content:space-between;color:var(--ink-2)}
.tip .r span:last-child{font-family:ui-monospace,Menlo,monospace;color:var(--ink)}
.hint{color:var(--ink-2);font-size:12.5px;margin:12px 0 0}
.hint code{font-family:ui-monospace,Menlo,monospace;font-size:12px;
background:var(--grid);padding:1px 4px;border-radius:3px}
.meta{color:var(--ink-3);font-size:11.5px;margin:16px 0 0;line-height:1.6}
</style></head><body><div class="wrap">
<h1>Specialist or generalist?</h1>
<p class="eyebrow">Federated synthetic EHR &middot; eICU &middot; __NSITES__ hospitals &middot; rare-code upweighting</p>
<div class="panel">
  <div class="controls">
    <div><label for="msel">Metric</label><select id="msel"></select></div>
    <div><label for="htog">Per site</label>
      <button id="htog" aria-pressed="false">hidden</button></div>
  </div>
  <div class="plotwrap">
    <svg id="p" viewBox="0 0 720 470" role="img"
      aria-label="Prevalence fidelity against each site's own test fold versus against the pooled cohort fold"></svg>
    <div class="tip" id="tip" role="status"></div>
  </div>
  <p class="hint" id="hint"></p>
  <p class="meta" id="meta"></p>
</div>
<script>
const D = __PAYLOAD__;
const $ = s => document.querySelector(s);
const cssv = n => getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();
const el = (t, a = {}) => { const e = document.createElementNS(
  "http://www.w3.org/2000/svg", t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const f3 = v => (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1)
  : v.toFixed(3);

function ticks(lo, hi, n = 6) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag)
    .find(s => s >= raw) || 10 * mag;
  const out = []; for (let t = Math.ceil(lo / step) * step; t <= hi; t += step)
    out.push(Math.abs(t) < step / 1e6 ? 0 : t);
  return out;
}

for (const k of D.metric_order)
  $("#msel").append(new Option(D.metrics[k].label, k));
$("#msel").value = D.metric_order[0];
$("#msel").addEventListener("change", draw);

let sites = false;
$("#htog").addEventListener("click", () => {
  sites = !sites;
  $("#htog").setAttribute("aria-pressed", String(sites));
  $("#htog").textContent = sites ? "shown" : "hidden";
  draw();
});

function draw() {
  const key = $("#msel").value, m = D.metrics[key];
  const marks = sites ? D.per_site[key] : [];
  const svg = $("#p"); svg.textContent = "";
  const W = 720, H = 470, P = { t: 26, r: 30, b: 62, l: 86 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const all = m.arms.concat(marks);
  const lo0 = Math.min(...all.map(p => Math.min(p.own, p.pooled)));
  const hi0 = Math.max(...all.map(p => Math.max(p.own, p.pooled)));
  const sp = (hi0 - lo0) || Math.abs(hi0) || 1;
  const lo = lo0 - sp * 0.1, hi = hi0 + sp * 0.1;
  const sx = v => P.l + (v - lo) / (hi - lo) * iw;
  const sy = v => P.t + ih - (v - lo) / (hi - lo) * ih;
  const g = el("g"); svg.append(g);
  const GRID = cssv("--grid"), AXIS = cssv("--axis"), MUT = cssv("--ink-3"),
        SEC = cssv("--ink-2"), PRI = cssv("--ink"), SURF = cssv("--surface");

  // The specialist half, tinted so the two sides of the diagonal read as
  // regions rather than as a line someone has to mentally reflect across.
  g.append(el("path", { d: `M${sx(lo)},${sy(lo)} L${sx(hi)},${sy(hi)} `
    + `L${sx(hi)},${sy(hi)} L${sx(lo)},${sy(hi)} Z`,
    fill: "#8fb6de", "fill-opacity": 0.07 }));
  for (const t of ticks(lo, hi)) {
    if (t < lo || t > hi) continue;
    g.append(el("line", { x1: P.l, x2: P.l + iw, y1: sy(t), y2: sy(t),
      stroke: GRID, "stroke-width": 1 }));
    g.append(el("line", { x1: sx(t), x2: sx(t), y1: P.t, y2: P.t + ih,
      stroke: GRID, "stroke-width": 1 }));
    let a = el("text", { x: P.l - 11, y: sy(t) + 4, "text-anchor": "end",
      "font-size": 11, fill: MUT,
      "font-family": "ui-monospace,Menlo,monospace" });
    a.textContent = f3(t); g.append(a);
    let b = el("text", { x: sx(t), y: P.t + ih + 21, "text-anchor": "middle",
      "font-size": 11, fill: MUT,
      "font-family": "ui-monospace,Menlo,monospace" });
    b.textContent = f3(t); g.append(b);
  }
  g.append(el("line", { x1: P.l, x2: P.l + iw, y1: P.t + ih, y2: P.t + ih,
    stroke: AXIS, "stroke-width": 1 }));
  g.append(el("line", { x1: P.l, x2: P.l, y1: P.t, y2: P.t + ih,
    stroke: AXIS, "stroke-width": 1 }));
  g.append(el("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi),
    stroke: cssv("--diag"), "stroke-width": 1.5, "stroke-dasharray": "5 4" }));

  let up = el("text", { x: P.l + 12, y: P.t + 17, "font-size": 10.5,
    fill: MUT, "font-style": "italic" });
  up.textContent = "above: fits its own site better — specialist";
  g.append(up);
  let dn = el("text", { x: P.l + iw - 10, y: P.t + ih - 11,
    "text-anchor": "end", "font-size": 10.5, fill: MUT,
    "font-style": "italic" });
  dn.textContent = "below: fits the cohort better — generalist";
  g.append(dn);

  let xt = el("text", { x: P.l + iw / 2, y: H - 16, "text-anchor": "middle",
    "font-size": 12, fill: SEC });
  xt.textContent = m.label + " — vs the POOLED cohort fold"; g.append(xt);
  let yt = el("text", { x: 20, y: P.t + ih / 2, "text-anchor": "middle",
    "font-size": 12, fill: SEC,
    transform: `rotate(-90 20 ${P.t + ih / 2})` });
  yt.textContent = m.label + " — vs the site's OWN fold"; g.append(yt);

  marks.forEach(p => { p.cx = sx(p.pooled); p.cy = sy(p.own);
    g.append(el("circle", { cx: p.cx, cy: p.cy, r: 3.4, fill: p.colour,
      "fill-opacity": 0.5, stroke: SURF, "stroke-width": 1 })); });
  m.arms.forEach(a => { a.cx = sx(a.pooled); a.cy = sy(a.own);
    g.append(el("circle", { cx: a.cx, cy: a.cy, r: 7.5, fill: a.colour,
      stroke: SURF, "stroke-width": 2 }));
    let t = el("text", { x: a.cx, y: a.cy - 15, "text-anchor": "middle",
      "font-size": 12, "font-weight": 600, fill: PRI,
      "font-family": "ui-monospace,Menlo,monospace" });
    t.textContent = a.label; g.append(t); });

  m.arms.concat(marks).forEach(p => {
    const hit = el("circle", { cx: p.cx, cy: p.cy, r: p.site ? 9 : 15,
      fill: "transparent", style: "cursor:pointer", tabindex: 0 });
    hit.setAttribute("role", "img");
    hit.setAttribute("aria-label", `${p.label}${p.site ? " at site " + p.site
      : ""}. own fold ${f3(p.own)}, pooled fold ${f3(p.pooled)}.`);
    const show = () => {
      const tip = $("#tip");
      const row = (k, v) => `<div class="r"><span>${k}</span><span>${v}</span></div>`;
      tip.innerHTML = `<b>${p.label}${p.site ? " @ " + p.site : ""}</b>`
        + row("own fold", f3(p.own)) + row("pooled fold", f3(p.pooled))
        + row("own − pooled", f3(p.own - p.pooled));
      tip.style.opacity = 1;
      const wrap = svg.parentElement.getBoundingClientRect(),
            box = svg.getBoundingClientRect();
      const cl = (v, a, b) => Math.max(a, Math.min(b, v));
      const cx = box.left - wrap.left + (p.cx / W) * box.width,
            cy = box.top - wrap.top + (p.cy / H) * box.height;
      tip.style.left = cl(cx + 15, 4,
        Math.max(4, wrap.width - tip.offsetWidth - 4)) + "px";
      tip.style.top = cl(cy - tip.offsetHeight / 2, 4,
        Math.max(4, wrap.height - tip.offsetHeight - 4)) + "px";
    };
    hit.addEventListener("mouseenter", show);
    hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", () => { $("#tip").style.opacity = 0; });
    hit.addEventListener("blur", () => { $("#tip").style.opacity = 0; });
    g.append(hit);
  });

  const up_ = m.better_up;
  const spec = m.arms.filter(a => up_ ? a.own > a.pooled : a.own < a.pooled);
  const gen = m.arms.filter(a => up_ ? a.own < a.pooled : a.own > a.pooled);
  const c = xs => xs.map(a => `<code>${a.label}</code>`).join(", ");
  $("#hint").innerHTML =
    `Both axes are <b>${m.label}</b>; only the real fold each arm is scored `
    + `against differs, so the dashed diagonal is where a generator matches its `
    + `own site exactly as well as it matches the federation. `
    + (spec.length ? `<b>Specialists</b> (above): ${c(spec)}. ` : "")
    + (gen.length ? `<b>Generalists</b> (below): ${c(gen)}. ` : "")
    + `The arms do not merely differ in quality &mdash; they reverse order `
    + `between the two targets, which is the personalization/generalization `
    + `frontier made visible on a density-modelling task.`
    + (sites ? " Small dots are individual sites." : "");
  $("#meta").innerHTML =
    `Points are the median over ${D.meta.n_sites} sites. Cohort `
    + `<code>${D.meta.cohort}</code>, ${D.meta.fold} fold, rare scope `
    + `<code>${D.meta.rare_scope}</code>, synthetic capped at `
    + `${D.meta.synth_cap} records per site so prevalence resolves to the same `
    + `1/N for every arm. Sites are relabelled H1..H${D.meta.n_sites} by `
    + `descending training size; no hospital identifier is carried here.`;
}
draw();
</script></div></body></html>
"""


def write_html(payload: dict, path: str) -> None:
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__NSITES__", str(payload["meta"]["n_sites"]))
    with open(path, "w") as fh:
        fh.write(html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=DEFAULTS["spec"],
                    help="test1 JSON written with --real-scope hospital")
    ap.add_argument("--gen", default=DEFAULTS["gen"],
                    help="test1 JSON written with --real-scope pooled")
    ap.add_argument("--sizes", default=DEFAULTS["sizes"],
                    help="a run JSON, read only for its partition sizes so the "
                         "H1..Hn relabelling is by descending site size")
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--stem", default=DEFAULTS["stem"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    spec, gen = load(args.spec, args.gen)
    payload = build(spec, gen, _site_order(spec, args.sizes))
    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, args.stem + ".png")
    htm = os.path.join(args.out_dir, args.stem + ".html")
    write_png(payload, png, args.dpi)
    write_html(payload, htm)

    print(f"cohort {payload['meta']['cohort']}  "
          f"({payload['meta']['condition']}), "
          f"{payload['meta']['n_sites']} sites, cap "
          f"{payload['meta']['synth_cap']}")
    for key in PNG_PANELS:
        m = payload["metrics"][key]
        print(f"\n  {m['label']}")
        print(f"    {'arm':13}{'own':>9}{'pooled':>9}{'own-pooled':>12}   side")
        for a in m["arms"]:
            d = a["own"] - a["pooled"]
            side = ("specialist" if (d > 0) == m["better_up"] else "generalist")
            print(f"    {a['label']:13}{a['own']:9.3f}{a['pooled']:9.3f}"
                  f"{d:12.3f}   {side}")
    print(f"\nwrote {png}\n      {htm}")


if __name__ == "__main__":
    main()
