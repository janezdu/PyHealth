"""Generate synthetic patients from a finished run's checkpoints.

Training and generation are different jobs with wildly different costs. Training
a HALO generator over the frozen 8-hospital cohort takes ~18 hours on one A100;
sampling 5,000 patients out of the trained model takes under two minutes. Baking
both into one script means every change to *how much* synthetic data you want --
a knob that only affects sampling -- drags an 18-hour GPU job behind it.

This module owns the sampling half. Point it at a finished run's ``save_dir``
and it rebuilds each generator from its ``.pt`` checkpoint, samples, and writes
``synthetic.json`` in the layout Test 1 and Test 2 already consume. Re-sizing a
run's synthetic output becomes a ~13-minute job instead of a re-train.

    python examples/fedpyhealth/generate.py \\
        --save-dir _outputs/local_full_hilo8_random_save \\
        --synth-per-hospital 4000

What lives where, per regime:

    regime        checkpoints on disk           generators
    ------------  ----------------------------  ----------------------------
    centralized   centralized_state.pt          1 (pooled)
    fedavg        fedavg_state.pt               1 (federated global)
    fedavg_ft     fedavg_state.pt + ft_<h>.pt   1 global + 8 fine-tuned
    local         local_<h>.pt                  8 independent

``fedavg_ft``'s ``ft_<h>.pt`` files are written by ``finetune_local``. Runs that
finished before that existed have only the pre-fine-tuning global, and this
script says so rather than silently generating from the wrong model.

Nothing here trains, and nothing here scores. Reading a checkpoint requires the
cohort cache only to rebuild the code vocabulary -- the generators are
unconditional, so no real patient data reaches the sampler.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch

from utils.cohort import DEFAULT_CACHE_DIR, load_fold, load_manifest
from pyhealth.models import HALO

# Checkpoint filename per regime. Single-model regimes name one file; the
# multi-model regimes format in the hospital id.
GLOBAL_CKPT = {
    "centralized": "centralized_state.pt",
    "fedavg": "fedavg_state.pt",
    "fedavg_ft": "fedavg_state.pt",
}
PER_HOSPITAL_CKPT = {
    "local": "local_{hid}.pt",
    "fedavg_ft": "ft_{hid}.pt",
}
# Regimes whose per-hospital synthetic data comes from one shared generator.
SINGLE_MODEL = ("centralized", "fedavg")


def load_run_config(save_dir: str, run_name: str = None) -> dict:
    """Read the architecture knobs a run was trained with.

    Rebuilding a generator needs the exact architecture its weights were
    written for; guessing gives a shape mismatch at best and a silently wrong
    model at worst. ``train.py`` stamps ``<save_dir>/config.json`` before
    training starts, so it exists even for a run that timed out with usable
    checkpoints. The results JSON is the fallback for runs predating that.

    Args:
        save_dir: The run's ``_outputs/<run_name>_save/`` folder.
        run_name: Overrides the name inferred from ``save_dir``.

    Returns:
        The run's config dict.

    Raises:
        FileNotFoundError: If neither source exists.
    """
    if run_name is None:
        base = os.path.basename(os.path.normpath(save_dir))
        run_name = base[:-5] if base.endswith("_save") else base

    direct = os.path.join(save_dir, "config.json")
    if os.path.exists(direct):
        with open(direct) as fh:
            return json.load(fh)

    legacy = os.path.join("_outputs", "results", "runs", f"{run_name}.json")
    if os.path.exists(legacy):
        with open(legacy) as fh:
            payload = json.load(fh)
        return payload.get("config", payload)

    raise FileNotFoundError(
        f"neither {direct} nor {legacy} exists, so the architecture this run "
        "was trained with is unknown and its checkpoints cannot be rebuilt. "
        "config.json is written at the start of train.py."
    )


def build_generator(cfg: dict, sample_dataset, save_dir: str):
    """Construct an untrained HALO matching a run's architecture.

    Mirrors ``train.py``'s ``build_model``. ``epochs`` is irrelevant here --
    nothing is trained -- but HALO takes it at construction, so it is pinned to
    1 rather than left to a default that might differ across versions.

    ``latent_dim`` matters the same way an adapter does, and for the same
    reason: the latent projection ``z_proj`` HAS parameters, so a model built
    without it cannot load a checkpoint trained with it. Both are read back out
    of the run's own config.json rather than passed in by the caller.

    If the run used an adapter, the adapter is re-applied here BEFORE any
    checkpoint is loaded. ``apply_adapter`` wraps modules, which renames every
    parameter beneath them (``attn.c_attn.weight`` becomes
    ``attn.c_attn.base.weight``, plus new ``A``/``B`` entries), so a plain HALO
    cannot accept an adapted run's state dict -- it fails with a wall of missing
    and unexpected keys. Reading the variant back out of the run's own
    config.json is what keeps the two sides in step without the caller having to
    remember which adapter a directory holds.
    """
    model = HALO(
        dataset=sample_dataset,
        embed_dim=cfg["embed_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_ctx=cfg["n_ctx"],
        batch_size=cfg["batch_size"],
        latent_dim=cfg.get("latent_dim", 0),
        dropout=cfg.get("dropout", 0.0),
        epochs=1,
        lr=cfg["lr"],
        save_dir=save_dir,
    )
    adapter = cfg.get("adapter", "none")
    if adapter and adapter != "none":
        from pyhealth.models.generators.adapters import apply_adapter
        apply_adapter(model.halo_model, adapter, int(cfg.get("adapter_rank", 8)))
    return model


def load_weights(model, path: str, expect_fingerprint: dict = None):
    """Load one checkpoint into ``model``, refusing a mismatched partition.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If the checkpoint was written for a different cohort or
            split than the one being generated against.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generation needs the trained weights; re-run "
            "that regime, or check that --save-dir points at a finished run."
        )
    ckpt = torch.load(path, map_location="cpu")
    if (expect_fingerprint is not None
            and ckpt.get("fingerprint") is not None
            and ckpt["fingerprint"] != expect_fingerprint):
        raise ValueError(
            f"{path} was written for a different partition or cohort than the "
            "one you are generating against; refusing to use it."
        )
    model.load_state_dict(ckpt["global_state"])
    return model


def proportional_pool(
    per_hospital: Dict[str, List[dict]], sizes: Dict[str, int], num_synth: int,
) -> List[dict]:
    """Subsample the per-hospital sets down to a real-cohort case mix.

    Pooled prevalence is a hospital-weighted average, so the only pooled view
    comparable against real pooled data is one mixed the way the real cohort is
    mixed. The uniform view (every hospital equal) is the concatenation of the
    per-hospital sets and is recovered downstream; it is a representation view,
    not a fidelity metric.

    Raises:
        ValueError: If a hospital generated fewer patients than its share.
    """
    total = float(sum(sizes.values())) or 1.0
    pooled: List[dict] = []
    for hid in per_hospital:
        share = max(1, int(round(num_synth * sizes[hid] / total)))
        if share > len(per_hospital[hid]):
            raise ValueError(
                f"hospital {hid} needs {share} patients for the proportional "
                f"pooled set but only {len(per_hospital[hid])} were generated. "
                f"Raise --synth-per-hospital to at least {share}, or lower "
                "--num-synth."
            )
        pooled.extend(per_hospital[hid][:share])
    return pooled


def generate_run(
    save_dir: str,
    regime: str,
    cfg: dict,
    sample_dataset,
    hospitals: List[str],
    sizes: Dict[str, int],
    num_synth: int,
    synth_per_hospital: int,
    device: str = "cpu",
    temperature: float = 1.0,
    fingerprint: dict = None,
    log=print,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Sample one finished run into ``(pooled_proportional, per_hospital)``.

    Single-model regimes emit one set and file it under every hospital key --
    the same object, because there is genuinely only one generator. Multi-model
    regimes give every hospital ``synth_per_hospital`` patients regardless of
    how much real data it holds, which is the whole point: a 79-patient site
    should be able to draw large amounts of synthetic data from a
    collaboratively trained generator.
    """
    if regime in SINGLE_MODEL:
        model = build_generator(cfg, sample_dataset, save_dir)
        load_weights(model, os.path.join(save_dir, GLOBAL_CKPT[regime]),
                     fingerprint)
        # One generator serves every hospital, so its output has no natural
        # per-site ceiling. Size it as the federation total (n_hospitals x the
        # per-hospital count) so a single pooled system and an 8-model
        # federation are compared at the same magnitude of synthetic data.
        want = max(num_synth, len(hospitals) * synth_per_hospital)
        log(f"[{regime}] one global generator -> {want} patients "
            f"({len(hospitals)} x {synth_per_hospital} federation total)")
        full = model.generate(num_samples=want, device=device,
                              temperature=temperature)
        # The pooled set stays num_synth for every regime, so Test 1's
        # prevalence resolution is identical across arms.
        return full[:num_synth], {hid: full for hid in hospitals}

    if regime not in PER_HOSPITAL_CKPT:
        raise ValueError(
            f"unknown regime {regime!r}; expected one of "
            f"{sorted(set(GLOBAL_CKPT) | set(PER_HOSPITAL_CKPT))}"
        )

    per_hospital: Dict[str, List[dict]] = {}
    for hid in hospitals:
        path = os.path.join(save_dir, PER_HOSPITAL_CKPT[regime].format(hid=hid))
        if regime == "fedavg_ft" and not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. This run predates fine-tuned-model "
                "checkpointing, so its 8 personalized generators exist nowhere "
                "on disk -- only the pre-fine-tuning global does. Re-run the "
                "fedavg_ft regime (it resumes the FedAvg half from "
                "fedavg_state.pt, so only fine-tuning repeats)."
            )
        model = build_generator(cfg, sample_dataset, save_dir)
        load_weights(model, path, fingerprint)
        per_hospital[hid] = model.generate(
            num_samples=synth_per_hospital, device=device,
            temperature=temperature)
        log(f"[{regime}] hospital {hid}: generated "
            f"{len(per_hospital[hid])} synthetic")

    pooled = proportional_pool(per_hospital, sizes, num_synth)
    log(f"[{regime}] pooled_proportional: {len(pooled)}   "
        f"uniform view: {sum(len(v) for v in per_hospital.values())}")
    return pooled, per_hospital


def write_synthetic(save_dir: str, pooled: List[dict],
                    per_hospital: Dict[str, List[dict]],
                    shared_generator: bool, suffix: str = "") -> str:
    """Write ``synthetic.json`` in the layout Test 1 and Test 2 read.

    ``pooled`` is stored twice under two names: ``pooled_proportional`` says
    which mixing rule produced it, and ``pooled`` is the pre-rename alias kept
    so older readers and already-finished runs stay valid. The uniform view is
    deliberately not stored -- it is exactly the concatenation of
    ``per_hospital``, so writing it would duplicate every record on disk.

    ``shared_generator`` records whether one model produced every hospital's
    set. It is written rather than inferred because inference is unreliable:
    each generator numbers its output ``synthetic_0..N`` independently, so eight
    genuinely different fine-tuned models emit eight identical *id* lists over
    completely different patients. A reader comparing ids would call that one
    shared set and collapse eight per-hospital classifiers into one.
    """
    path = os.path.join(save_dir, f"synthetic{suffix}.json")
    with open(path, "w") as fh:
        json.dump({"pooled": pooled,
                   "pooled_proportional": pooled,
                   "shared_generator": bool(shared_generator),
                   "per_hospital": per_hospital}, fh)
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-dir", required=True,
                   help="the run's _outputs/<run_name>_save/ folder, holding "
                        "its .pt checkpoints")
    p.add_argument("--run-name",
                   help="overrides the run name inferred from --save-dir, used "
                        "to find the results JSON holding the architecture")
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR,
                   help="cohort cache the run was trained on; needed only to "
                        "rebuild the code vocabulary")
    p.add_argument("--num-synth", type=int,
                   help="size of the proportional pooled set (default: the "
                        "value the run was trained with)")
    p.add_argument("--synth-per-hospital", type=int,
                   help="patients per hospital for the multi-model regimes, "
                        "identical for every hospital regardless of its real "
                        "size (default: the run's own value)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="logit temperature at generation, applied before the "
                        "sigmoid. The codes are conditionally independent "
                        "Bernoullis, so expected codes/visit = sum of their "
                        "probabilities -- temperature is therefore a direct "
                        "dial on emission volume with NO retraining. <1 sharpens "
                        "and emits fewer codes, >1 flattens and emits more; 1.0 "
                        "(default) is the trained model. Useful for asking how "
                        "much of a prevalence gap is calibration rather than "
                        "representation. Writes to a suffixed synthetic file so "
                        "a swept run never overwrites the tau=1 baseline.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be generated, touching no weights")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_run_config(args.save_dir, args.run_name)
    regime = cfg["regime"]
    num_synth = args.num_synth or cfg["num_synth"]
    per_hosp_n = args.synth_per_hospital or cfg.get("synth_per_hospital")
    if per_hosp_n is None:
        # Runs predating this knob conflate the two roles: their num_synth is
        # both the pooled size and the per-hospital size. Falling back to it
        # silently produces n_hospitals x num_synth patients -- 40k for a
        # 5000-patient run -- which looks like a successful job and only
        # surfaces as a wrong per-arm training-set size much later. Refuse for
        # EVERY regime; the caller knows what they want, this file does not.
        raise SystemExit(
            f"{args.save_dir} was trained before --synth-per-hospital existed, "
            f"so its config cannot say how many patients each hospital should "
            f"get (its num_synth={num_synth} is the POOLED size, not the "
            f"per-hospital one). Pass --synth-per-hospital explicitly."
        )

    manifest = load_manifest(args.cohort_cache)
    hospitals = [str(h) for h in manifest["hospitals"]]
    sizes = {h: manifest["per_hospital"][h]["n_train"] for h in hospitals}

    print(f"run       : {cfg.get('run_name')}")
    print(f"regime    : {regime}"
          + ("  (one generator)" if regime in SINGLE_MODEL
             else f"  ({len(hospitals)} generators)"))
    print(f"num_synth : {num_synth}   synth_per_hospital: {per_hosp_n}")
    print(f"device    : {args.device}")
    if args.dry_run:
        want = (num_synth if regime in SINGLE_MODEL
                else per_hosp_n * len(hospitals))
        print(f"dry run: would sample {want} patients total; no weights read.")
        return

    # The cache is read only to recover the shared code vocabulary -- the
    # generators are unconditional and never see a real patient here.
    sample_dataset = load_fold(args.cohort_cache, "train")
    fingerprint = None  # accept any partition; the checkpoint carries its own

    pooled, per_hospital = generate_run(
        args.save_dir, regime, cfg, sample_dataset, hospitals, sizes,
        num_synth, per_hosp_n, device=args.device, fingerprint=fingerprint,
        temperature=args.temperature,
    )
    # A swept temperature writes to its own file. generate.py overwrites
    # synthetic.json in place, so without this a tau sweep would destroy the
    # tau=1 synthetic set that every existing score was computed from.
    suffix = "" if args.temperature == 1.0 else f"_tau{args.temperature:g}"
    path = write_synthetic(args.save_dir, pooled, per_hospital,
                           shared_generator=regime in SINGLE_MODEL,
                           suffix=suffix)
    print(f"\nSaved synthetic data -> {path}")
    print("Re-score it with test1_prevalence.py / test2_rare_efficacy.py; "
          "neither needs a GPU.")


if __name__ == "__main__":
    main()
