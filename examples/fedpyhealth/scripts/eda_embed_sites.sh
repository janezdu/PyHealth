#!/bin/bash
# Where do a hospital's synthetic patients sit, relative to its real ones?
#
# ONE figure: a row per hospital, a column per regime. Rows are the SMALLEST
# site (429, 142 train patients) and the LARGEST (458, 1832) -- the size axis
# is the point, since if federation helps anyone it is meant to be 429, and
# 458 is the site whose data dominates the average. Every panel shares one
# projection fitted on the real pooled train split, so rows and columns are
# both comparable.
#
# The companion eda_code_embedding.py (what code geometry each arm LEARNT, out
# of HALO's vis_embed_mat) is deliberately NOT run here -- the patient view is
# the one being carried forward. Run that script directly if you want it back.
#
# CPU-only, a couple of minutes. No model is loaded: every arm's
# synthetic.json is already on disk, so no GPU and no torch.
#
#   srun --account=bgyw-delta-cpu --partition=cpu-interactive --nodes=1 \
#        --ntasks-per-node=1 --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   bash examples/fedpyhealth/scripts/eda_embed_sites.sh
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE COMPARING THE LAST TWO COLUMNS. fedavg_irm and
# fedavg_irm_xm share rho=100 and warmup=17, but the xm run ALSO carries
# dropout 0.1 and batch 256 against the others' dropout 0 and batch 128:
#
#   arm             irm_rho  warmup  xm_k  dropout  batch
#   fedavg              -       -      -      0      128
#   fedavg_irm        100      17      -      0      128
#   fedavg_irm_xm     100      17      4     0.1     256
#
# Dropout is not incidental -- notes/ records that best-of-K needs a source of
# spread or the K passes are bit-identical and it is best-of-1 at K x the cost.
# Batch 256 is the loose end. So a gap between those two columns is
# "xm + dropout + batch", not "xm". Two more replicates of the same config
# exist (varrep2_, varrep3_ ...irm100w17_do0.1_xm4) if you want a noise floor
# before reading anything into it.
# ---------------------------------------------------------------------------
#
# Knobs: HOSPITALS=429,358,458 (comma-separated), COHORT=lo8_random, POOL=rare,
# METHOD=tsne, FIT_ON=stack, IRM_RUN / XM_RUN to swap runs.
set -euo pipefail

HOSPITALS="${HOSPITALS:-429,458}"
COHORT="${COHORT:-hilo8_random}"
POOL="${POOL:-scored}"
METHOD="${METHOD:-pca}"
FIT_ON="${FIT_ON:-real_pooled}"
# warmup 17 so it matches the xm arm on rho AND warmup; the nes_...w68 run is
# the other tuned option but then two things differ across that column pair.
IRM_RUN="${IRM_RUN:-fedavg_full_${COHORT}_irmrho100}"
XM_RUN="${XM_RUN:-fedavg_E2_R50_${COHORT}_privacy_irm100w17_nes_do0.1_xm4}"
CACHE="${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the cohort cache root}/${COHORT}"
OUTDIR="${OUTDIR:-_outputs/results}"

[ -d .venv ] && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=

LOCAL_DIR="_outputs/local_full_${COHORT}_nes_save"
FEDAVG_DIR="_outputs/fedavg_full_${COHORT}_nes_save"
IRM_DIR="_outputs/${IRM_RUN}_save"
XM_DIR="_outputs/${XM_RUN}_save"
CENTRAL_DIR="_outputs/centralized_full_${COHORT}_nes_save"

echo "IRM arm:    ${IRM_DIR}"
echo "IRM+xm arm: ${XM_DIR}"
for d in "$LOCAL_DIR" "$FEDAVG_DIR" "$IRM_DIR" "$XM_DIR" "$CENTRAL_DIR"; do
    [ -f "$d/synthetic.json" ] || {
        echo "missing $d/synthetic.json" >&2; exit 1; }
done

python -u examples/fedpyhealth/scripts/eda_embed_projection.py \
    --cohort-cache "$CACHE" \
    --hospitals "$HOSPITALS" \
    --fold test --code-pool "$POOL" --method "$METHOD" --fit-on "$FIT_ON" \
    --n-synth 400 \
    --run local="$LOCAL_DIR" \
    --run fedavg="$FEDAVG_DIR" \
    --run fedavg_irm="$IRM_DIR" \
    --run fedavg_irm_xm="$XM_DIR" \
    --run centralized="$CENTRAL_DIR" \
    --out "${OUTDIR}/eda_embed_grid"

echo
echo "done. wrote:"
ls -la "${OUTDIR}/eda_embed_grid".{json,png}
