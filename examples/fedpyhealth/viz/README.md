# Fidelity vs utility — federated synthetic EHR

A self-contained interactive scatter of the four training regimes
(`local`, `fedavg_ft`, `fedavg`, `centralized`) on two axes: prevalence
fidelity (Test 1) against rare-code TSTR (Test 2). Both axes are selectable,
so any of the 6 prevalence metrics can be plotted against any of the 5 TSTR
metrics, with the `prior` / `real_local` / `real_pooled` baselines drawn as
reference lines.

## Viewing it

`index.html` has no external dependencies — no CDN, no fonts, no separate JS.
Open it in any browser:

```bash
open examples/fedpyhealth/viz/index.html          # macOS
xdg-open examples/fedpyhealth/viz/index.html      # Linux
```

**On a cluster over SSH**, serve it and let your editor forward the port:

```bash
python3 -m http.server 8000 --directory examples/fedpyhealth/viz
```

**Do not use an editor's built-in HTML preview pane.** Those render in a webview
with a Content-Security-Policy that blocks inline `<script>`, so the CSS applies
but the chart and the dropdowns come up empty. It looks broken; it isn't.

## What's in it

`data.json` is the extracted source data, regenerated from the two results
files by the snippet in "Regenerating" below. The same object is inlined into
`index.html` so the page stays self-contained; `data.json` is kept alongside it
for anyone who wants the numbers without parsing HTML.

Values are macro-averaged across the 8 hospitals. Nothing patient-level, no
hospital identifiers, no paths — safe to commit.

## Two caveats, also printed on the page

1. **The TSTR axis is measuring a downstream task that is not working.** Every
   arm scores below chance on AUROC (0.5) and at or below the `prior` floor on
   recall@10 — inverted, not merely weak. Leading hypothesis is a masking
   artifact: a patient carrying more rare codes has more stripped from their
   input, so the model reads them as low-risk.
2. **The two axes are not matched the same way.** Prevalence is capped at 2,000
   synthetic patients per hospital for every regime (`--synth-cap 2000`). TSTR
   is uncapped, so `centralized` and `fedavg` trained on 16,000 records against
   `local` and `fedavg_ft`'s 2,000 — part of the vertical spread is data volume,
   not generator quality.

## Regenerating

After a fresh `test1_prevalence.py` / `test2_rare_efficacy.py` run, rebuild
`data.json` and paste it over the `const DATA = ...` line in `index.html`.
The page reads nothing at runtime, so that one line is the only thing to update.

Source files:

- `_outputs/results/tests/test1_4arm_cap2000.json`
- `_outputs/results/tests/test2_4arm_uncapped.json`

## Design notes

Colour is an **ordinal** blue ramp, not categorical hues — the regimes have a
natural order (isolated → fully pooled) and that order is the story. It also
sidesteps a hard constraint: a scatter needs all-pairs colour validation, and
the default 4-slot categorical palette fails it (yellow↔orange normal-vision
ΔE 13.7, below the 15 floor). The ordinal ramp passes every check in both light
and dark mode.

Every point carries a direct label *and* appears in the legend, so identity
never depends on colour alone; labels de-collide with leader lines back to
their own mark. A full table view sits below the chart.
