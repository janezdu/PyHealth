# Federated IRM for generative EHR — design note

**Status:** proposal, nothing implemented. Written 2026-08-26.

Adding the IRMv1 penalty to the FedAvg HALO run on `hilo8_random`, with each
hospital as an IRM *environment*.

---

## 1. The objective

$$\min_{\theta} \; \sum_{k=1}^{K} \Big[\, R_k(\theta) \;+\; \rho \cdot \big\|\nabla_{w \,|\, w=1.0}\, R_k(w \cdot \theta)\big\|^2 \,\Big]$$

$R_k$ is hospital $k$'s loss, $w$ a scalar dummy fixed at $1.0$, $\rho$ the
penalty weight.

The gradient term asks: *how far is $w=1$ from optimal for this environment?* If
$w=1$ is simultaneously optimal everywhere, the penalty is zero and the
representation is invariant.

### The equation as drawn is ambiguous — read it as scaling the output

$R_k(w \cdot \theta)$ literally reads as $w$ scaling the **parameters**. Standard
IRMv1 scales the **output logits** — $w$ is a dummy classifier stacked on the
representation, not a rescaling of the network. Use the standard reading.

Because $w$ is a scalar, $\nabla_w R_k$ is a **scalar**, and the penalty is that
scalar squared. Not a vector norm over parameters — which is the whole reason
IRMv1 is cheap.

---

## 2. Does it need extra communication? No.

The penalty sits **inside** the sum. It is per-environment, then summed — not the
penalty of the pooled loss. So the objective is **fully separable across
clients**: client $k$ computes both $R_k$ and $P_k = \|\nabla_w R_k\|^2$ from its
own data and the broadcast weights alone.

Under FedAvg, nothing about the protocol changes:

```
server broadcasts θ
  ↓
client k:  L_k = R_k(θ) + ρ · P_k(θ)      ← the only change: its local loss
           run local_epochs of SGD on L_k
  ↓
server: weighted average of client weights, as before
```

**Zero extra bytes required.** Your instinct that something extra must be
communicated is the natural one — it just isn't true for this particular
penalty, because IRMv1 is separable by construction.

### Send the two scalars anyway — for monitoring

$R_k$ and $P_k$ as **separate** values (your option b), not their sum. Two floats
per client per round, used for instrumentation rather than for the algorithm:

- $\rho$ needs a sweep and you cannot tune it without seeing the two terms'
  relative magnitude.
- The characteristic IRM failure is the penalty collapsing to zero by making the
  model useless. Only visible if $R$ and $P$ are logged apart.
- Per-hospital $P_k$ says *which* sites are the non-invariant ones — a result in
  itself, given this cohort's 13× size spread.

Sending only the sum throws that away for no saving.

---

## 3. Where it does get interesting: client drift

This is the part that isn't trivial, and where a contribution would live.

Centralized IRM evaluates every $\nabla_w R_k$ at the **same** $\theta$. FedAvg
does not — over `local_epochs=2`, client $k$ drifts to its own $\theta_k$ before
its penalty is ever computed. Each client answers:

> *"Is $w=1$ optimal for me, at **my own drifted** parameters?"*

rather than the question IRM actually asks:

> *"Is $w=1$ optimal for me, at the **shared** parameters?"*

The drifted version is weaker, and weaker in the damaging direction: local drift
lets each client partially re-fit its own environment, which is exactly what IRM
exists to prevent.

### Three ways to handle it

| approach | cost | notes |
|---|---|---|
| **`local_epochs=1`** | free | Closest to centralized IRM. Keep the budget matched at `n_rounds=100 × local_epochs=1`. |
| **Penalty at the broadcast $\theta$** | 1 extra fwd/bwd per client per round | Compute $\nabla_w R_k$ once at $\theta$ before any local step, hold it fixed through the local epochs. SCAFFOLD-flavored correction. |
| **`centralized` + IRM** | one training job | Environments still hospitals, data pooled, no drift at all. The oracle. |

Start with **rows 1 and 3**. Row 3 separates two different questions:

- *Does IRM help on this problem at all?* → centralized + IRM
- *Does FedAvg approximate IRM well?* → the gap between federated and centralized

If centralized-IRM shows nothing, federated-IRM will not either, and you have
saved yourself the harder implementation.

---

## 4. The conceptual wrinkle: invariance without a label

IRM assumes a stable causal mechanism $P(Y \mid \Phi(X))$ that holds across
environments, plus spurious correlations that do not. **HALO is unconditional —
there is no $Y$.**

The per-step prediction is next-code-given-history, so what you would be
enforcing is:

> the conditional code distribution given patient history is the same at every
> hospital.

But site-specific case mix is *precisely* what differs across these eight
hospitals, and Test 1 measures per-hospital prevalence fidelity. **You would be
penalizing the model for fitting the thing one of your two metrics rewards.**

That is not a reason to skip it. It is the hypothesis:

> **IRM should hurt the specialist target and help the generalist target.**

| target | `--real-scope` | prediction under IRM |
|---|---|---|
| each site vs its own test fold | `hospital` | **worse** |
| all sites vs the pooled cohort fold | `pooled` | **better** |

Your head-vs-tail panel already plots these two against each other. If the
prediction holds cleanly, that is a real result about what invariance buys in
generative EHR. If both move the same way, something more interesting — or more
broken — is happening, and either is worth knowing.

### The rare-code angle sharpens it

In IRM's vocabulary:

- a code **rare at one site but common elsewhere** is *spurious*
- a code **rare everywhere** is *invariant*

Your manifest already splits exactly these: 548 `pooled_rare_codes` against 475
`global_rare_codes` on hilo8_random. That is a pre-registered subgroup analysis
sitting there for free — IRM should help the globally-rare set and hurt the
locally-rare-but-globally-common one.

---

## 5. Implementation notes

### Where $w$ attaches

`pyhealth/models/generators/halo.py:325`:

```python
code_probs = sig(code_logits)          # →  sig(w * code_logits)
```

One line, on a function that already exists. The `sample_weight_fn` seam added
for rare-upweighting threads a per-batch hook through all four training helpers;
the penalty wants a similar one.

### The biased estimator — this is what sinks naive attempts

$$\|\mathbb{E}[g]\|^2 \;\neq\; \mathbb{E}\big[\|g\|^2\big]$$

Minibatch noise inflates the squared norm, and the model then minimizes *noise*
rather than achieving invariance. The IRM reference implementation splits each
batch in half and uses $g_1 \cdot g_2$ as an unbiased estimate of
$\|\mathbb{E}[g]\|^2$.

**This will be worst at the small sites.** Hospital 429 has 142 train patients at
`batch_size=128` — one batch per epoch — so a half-batch penalty estimate there
is close to pure noise. Expect to need a smaller batch for the IRM arm, which
then breaks comparability with every batch-128 result in the project. **Decide
that trade-off before launching, not after.**

### Penalty warmup is not optional

The paper runs $\rho \approx 1$ for a warmup period, then jumps to
$\rho \sim 10^4$, and rescales the whole loss by $\rho$ once it is large to stop
gradients exploding. Without warmup IRM commonly fails to fit at all — and you
would conclude the method does not work when it was never given a chance.

$\rho$ is the one hyperparameter that genuinely needs a sweep.

---

## 6. Proposed experiment

Four arms on `hilo8_random`, batch and budget matched to existing runs:

| arm | status |
|---|---|
| `fedavg` | **exists** |
| `fedavg_irm` | new |
| `centralized` | **exists** |
| `centralized_irm` | new |

Two training jobs plus a $\rho$ sweep. Score through
`scripts/score_cohort.sh` so it lands on the same axes as everything else — both
Test 1 targets, matched Test 2 budget, 8,000 per hospital.

### What would make this publishable rather than a knob

The federated-IRM-for-classification literature exists. Two things here are less
covered:

1. **IRM on an unconditional generative model**, where "invariance" has to be
   redefined because there is no label.
2. **The specialist/generalist trade-off measured directly** — most IRM papers
   report one number on held-out environments. You have both targets on the same
   plot for the same runs.

### What would kill it

- The penalty is too noisy at 84–142-patient sites to carry signal, and the IRM
  arm is indistinguishable from FedAvg at every $\rho$ that still fits the data.
- Invariance costs specialist fidelity without buying generalist fidelity —
  a null result, but a reportable one given the pre-registered prediction.

---

## Open questions

- Should the environment be the **hospital**, or something else? Hospitals are
  the obvious choice, but eICU also carries `region` and `teachingstatus` — a
  coarser environment split would have more data per environment and less penalty
  noise, at the cost of fewer environments. **Four regions instead of eight
  hospitals** might be the better-powered version of this experiment.
- Does the penalty want to be weighted by client size, the way FedAvg's
  aggregation is (`weighting: sample`)? The objective as written sums penalties
  unweighted, which gives hospital 429 (142 patients) the same voice as 458
  (1,832). That may be correct — IRM is about environments, not samples — but it
  should be a stated choice rather than an accident.
- `local_epochs=1` doubles `n_rounds` to hold the budget, which doubles
  communication. Fine here, worth noting for any claim about communication cost.
