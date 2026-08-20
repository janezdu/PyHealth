"""Freeze what ``HALO._encode_visits`` produces today, so a rewrite can prove
it changed nothing.

The multi-hot rewrite is a pure performance change: the tensors handed to the
transformer must come out bit-identical, or every checkpoint and every published
number silently stops being comparable. This script captures the current output
for real cohort batches; ``tests/core/test_halo_encode_golden.py`` replays it.

Run BEFORE touching the encoder::

    python examples/fedpyhealth/utils/capture_encode_golden.py \\
        --cohort-cache $FEDCOHORT_CACHE/hilo8_random

It also times the encode step against the rest of a training step, which is the
number the whole plan rests on -- if encoding is not the dominant cost, the
rewrite is not worth the blast radius.

CPU by default so the captured tensors are deterministic and portable. Pass
``--device cuda`` for the timing to mean anything, since the per-patient
``.item()`` sync only costs on a GPU.
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import torch

# This module lives in utils/ but imports utils.cohort, so the example root has
# to be importable. Scripts like eda.py sit one level up and get that for free;
# a verification script run directly does not.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cohort import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    load_clients,
    load_processor,
)

GOLDEN = os.path.join("_outputs", "golden", "encode_golden.pt")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--hospital", default=None,
                   help="which hospital's train fold to sample (default: the "
                        "first in the manifest)")
    p.add_argument("--batches", type=int, default=3,
                   help="how many batches to capture")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu for a portable capture; cuda for meaningful timing")
    p.add_argument("--out", default=GOLDEN)
    p.add_argument("--time-only", action="store_true",
                   help="benchmark without writing a golden file")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    from pyhealth.datasets import get_dataloader
    from pyhealth.models import HALO
    from utils.cohort import hospitals

    hid = args.hospital or hospitals(args.cohort_cache)[0]
    clients = load_clients(args.cohort_cache, folds=("train",), only=[hid])
    ds = clients[hid]["train"]
    proc = load_processor(args.cohort_cache)
    # max_inner_len only exists on the legacy NestedSequenceProcessor; the
    # multi-hot form has no inner axis to pad, which is the point of the rewrite.
    inner = getattr(proc, "_max_inner_len", None)
    print(f"hospital {hid}: {len(ds)} patients, vocab {proc.vocab_size()}, "
          f"max_inner_len {inner if inner is not None else 'n/a (multi-hot)'}")

    # Geometry matching PROFILES["full"], so the captured tensors have the same
    # shape the real runs use.
    model = HALO(dataset=ds, embed_dim=256, n_heads=4, n_layers=4, n_ctx=50,
                 batch_size=args.batch_size, epochs=1, lr=1e-4)
    # HALO.device is a read-only property derived from the model's parameters,
    # so .to() is the only way to move it -- assigning to .device raises.
    model.to(args.device)

    loader = get_dataloader(ds, batch_size=args.batch_size, shuffle=False)
    captured: List[Dict[str, torch.Tensor]] = []
    enc_s, fwd_s, load_s, h2d_s = 0.0, 0.0, 0.0, 0.0

    # The loader produces `batch` before the loop body runs, so its cost is
    # invisible to any timer started inside the body. Measure it as the gap
    # since the previous iteration ended. get_dataloader leaves num_workers at
    # 0, so this is synchronous main-process work that the GPU waits on -- and
    # with encoding gone it is a candidate for the new bottleneck.
    first: Dict[str, float] = {}
    t_prev = time.perf_counter()
    for i, batch in enumerate(loader):
        t_arrived = time.perf_counter()
        if i >= args.batches:
            break
        visits = batch["visits"].to(args.device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ehr, mask = model._encode_visits(visits)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # The rest of the step: forward + backward, i.e. the actual learning.
        loss, _, _ = model.halo_model(
            ehr, position_ids=None, ehr_labels=ehr, ehr_masks=mask,
            pos_loss_weight=model.config.pos_loss_weight)
        loss.backward()
        if args.device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        model.halo_model.zero_grad(set_to_none=True)

        enc_s += t1 - t0
        fwd_s += t2 - t1
        load_s += t_arrived - t_prev
        h2d_s += t0 - t_arrived
        captured.append({"visits": visits.cpu(),
                         "batch_ehr": ehr.detach().cpu(),
                         "batch_mask": mask.detach().cpu()})
        print(f"  batch {i}: visits {tuple(visits.shape)} -> ehr "
              f"{tuple(ehr.shape)}   load {t_arrived - t_prev:6.3f}s   h2d "
              f"{t0 - t_arrived:6.3f}s   encode {t1 - t0:6.3f}s   fwd+bwd "
              f"{t2 - t1:6.3f}s")
        if i == 0:
            first = {"load (fetch+collate)": t_arrived - t_prev,
                     "h2d copy": t0 - t_arrived,
                     "encode": t1 - t0, "fwd+bwd": t2 - t1}
        t_prev = time.perf_counter()

    n = max(1, len(captured))
    parts = {"load (fetch+collate)": load_s, "h2d copy": h2d_s,
             "encode": enc_s, "fwd+bwd": fwd_s}
    total = sum(parts.values())

    def _report(label_s, skip_first):
        # Batch 0 carries CUDA context creation, cuDNN autotuning and the first
        # touch of every cache -- seconds of one-off cost that swamp a step now
        # measured in milliseconds. Steady state is the number that predicts an
        # epoch; both are printed so neither can be quoted out of context.
        d = max(1, n - 1) if skip_first else n
        print(f"\n{label_s} ({d} batch{'es' if d != 1 else ''} of "
              f"{args.batch_size} on {args.device}):")
        tot = 0.0
        for name, acc in parts.items():
            v = (acc - first.get(name, 0.0)) if skip_first else acc
            tot += v
        for name, acc in parts.items():
            v = (acc - first.get(name, 0.0)) if skip_first else acc
            pct = 100 * v / tot if tot else 0.0
            print(f"  {name:22s} {v / d:8.4f} s/batch  {pct:5.1f}%")
        print(f"  {'step total':22s} {tot / d:8.4f} s/batch")
        return tot / d

    _report("including batch 0 (with CUDA warmup)", False)
    if n > 1:
        step = _report("steady state, batch 0 excluded", True)
        slowest = max(parts, key=lambda k: parts[k] - first.get(k, 0.0))
        print(f"\n  -> the bottleneck is now: {slowest}")
        print(f"     at {step:.4f} s/batch, an epoch of B batches costs "
              f"~{step:.3f} * B seconds.")

    if args.time_only:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "batches": captured,
        "meta": {"cohort_cache": args.cohort_cache, "hospital": hid,
                 "batch_size": args.batch_size, "n_ctx": 50,
                 "vocab_size": proc.vocab_size(),
                 "max_inner_len": inner,
                 "total_vocab_size": model.config.total_vocab_size},
    }, args.out)
    print(f"\nwrote {args.out} ({len(captured)} batches) -- this is the contract "
          "the rewrite must match")


if __name__ == "__main__":
    main()
