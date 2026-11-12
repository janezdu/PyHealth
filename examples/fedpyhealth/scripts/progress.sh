#!/bin/bash
# What is queued, what is running, and how far along each running job is.
#
# Progress is read from the LOG, not from squeue: squeue only says RUNNING.
# test1_prevalence flushes a line per (arm, hospital) and train.py flushes a
# line per round, so counting those lines is the only honest progress signal.
cd "$(dirname "$0")/../../.." || exit 1

echo "== queue =="
squeue -u "$USER" -o "%.16i %.14j %.9T %.7M %.9l %R" 2>/dev/null

echo
echo "== running jobs, progress from their logs =="
squeue -u "$USER" -h -t RUNNING -o "%A %j" 2>/dev/null | while read -r id name; do
    log=$(ls -t _outputs/slurm/*"$id"*.out 2>/dev/null | head -1)
    [ -z "$log" ] && { echo "  $name ($id): no log yet"; continue; }
    arms=$(grep -c '^=== ' "$log" 2>/dev/null || echo 0)
    hosp=$(grep -c ' ->  all R2=' "$log" 2>/dev/null || echo 0)
    rounds=$(grep -c 'aggregated' "$log" 2>/dev/null || echo 0)
    printf "  %-16s (%s)  " "$name" "$id"
    [ "$rounds" -gt 0 ] && printf "rounds=%s  " "$rounds"
    [ "$arms" -gt 0 ] && printf "arms=%s hospitals=%s  " "$arms" "$hosp"
    printf "\n     last: %s\n" "$(tail -1 "$log" | cut -c1-96)"
done

echo
echo "== outputs that have landed =="
for pat in "test1_frontier1src_*" "test1_zwidth_*" "test1_xmfill_*"; do
    n=$(ls _outputs/results/tests/$pat.json 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "  $pat -> $n file(s)"
done
ls -d _outputs/dp_probe_*_save _outputs/do_lora_ft_save 2>/dev/null | sed 's|_outputs/|  trained: |'
exit 0
