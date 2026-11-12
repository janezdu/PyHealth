#!/bin/bash
# The upweighting x early-stopping 2x2, all three approaches, at 200 bootstraps.
#
# WHAT THIS FIXES. The appendix subsection "Rare-code upweighting" compares
# unweighted against upweighted for local-only, FedAvg and centralized, and its
# own Caveat paragraph concedes the comparison is unusable:
#
#   "The upweighted runs were trained with early stopping and the unweighted
#    runs for the full budget, and each condition was run once. The differences
#    above therefore reflect weighting and stopping together and cannot be
#    attributed to the weighting alone."
#
# dob9 cites exactly that paragraph (their A.3) as the reason the centralized
# reference -- and therefore the frontier it anchors -- cannot be interpreted.
# The upweighted centralized run early-stopped at 17 of 100 epochs, so the
# anchor got one-sixth the compute of everything measured against it.
#
# The missing cell in each row was upweighted + full budget. Array tasks 2, 9
# and 10 of _train_v2_allarms.sh supplied them (v2_centralized_rw,
# v2_local_rw, v2_fedavg_rw -- BASE carries --no-early-stop). This scores all
# twelve cells together so weighting and stopping can finally be read apart:
#
#                     early stopping        full budget
#   unweighted        <approach>_save       <approach>_nes_save
#   upweighted        <approach>_rw_save    v2_<approach>_rw_save
#
# CPU ONLY -- prevalence scoring does no GPU work. CUDA is masked so this
# cannot silently take a device.
#
# PREREQUISITE: _gen_v2_local_rw.sh, which brings the one under-generated cell
# up to 8000/hospital. Submit this with --dependency=afterok:<that job>.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=score-rwes-2x2
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

# cell name = approach _ weighting _ stopping. Kept short so the output table
# reads as a grid rather than as a list of run directories.
RUNS="--run cent_unw_es=_outputs/centralized_full_hilo8_random_save
      --run cent_unw_full=_outputs/centralized_full_hilo8_random_nes_save
      --run cent_rw_es=_outputs/centralized_full_hilo8_random_rw_save
      --run cent_rw_full=_outputs/v2_centralized_rw_save
      --run fed_unw_es=_outputs/fedavg_full_hilo8_random_save
      --run fed_unw_full=_outputs/fedavg_full_hilo8_random_nes_save
      --run fed_rw_es=_outputs/fedavg_full_hilo8_random_rw_save
      --run fed_rw_full=_outputs/v2_fedavg_rw_save
      --run loc_unw_es=_outputs/local_full_hilo8_random_save
      --run loc_unw_full=_outputs/local_full_hilo8_random_nes_save
      --run loc_rw_es=_outputs/local_full_hilo8_random_rw_save
      --run loc_rw_full=_outputs/v2_local_rw_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

echo "############ Test 1, SPECIALIST (own test fold), 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_rwes2x2_spec_b200.json

echo "############ Test 1, GENERALIST (pooled test fold), 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_rwes2x2_gen_b200.json
echo done
