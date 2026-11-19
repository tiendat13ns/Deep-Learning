"""RQ1, Final: robustness checks and error audit on top of the frozen thresholds.

Run from ati_final (after `python -m src.analyze_thresholds`):
    python -m src.rq1_audit --part bootstrap   # paired bootstrap over test comments
    python -m src.rq1_audit --part seeds       # calibrate + evaluate seeds 43/44 and aggregate
    python -m src.rq1_audit --part errors      # quantitative error tables + case sample
    python -m src.rq1_audit --part summary     # tally the manual audit in annotations.csv + error figures
    python -m src.rq1_audit                    # all four

The case sample is deterministic (seed 42), so rerunning --part errors rewrites
the same cases.csv that annotations.csv was written against.

Nothing here fits or changes the thresholds of the main run: it only reads
results/rq1/calibration.json and the decoded predictions written by
src.analyze_thresholds. Extra seeds get their own calibration (on their own
validation scores) in results/rq1/seeds/<run>/. No comment text is written to
disk: the case sample holds ids, labels and probabilities only. Train labels are
read only to compute train co-occurrence (phi), never test labels for that.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, SEED, label_matrix, load_split

ROOT = Path(__file__).resolve().parent.parent
RQ1 = ROOT / "results" / "rq1"
MAIN_RUN = "phobert_text_pw_sqrt_ep12"
EXTRA_SEEDS = (43, 44)
STRATEGIES = ("fixed", "global", "per_label")
COMPARISONS = (("global", "fixed"), ("per_label", "fixed"), ("per_label", "global"))
N_BOOT = 10_000


def _tiers():
    spec = json.loads((RQ1 / "calibration.json").read_text(encoding="utf-8"))
    return spec, np.array([spec["tiers"][label] for label in LABELS])


def load_decoded(run=MAIN_RUN, split="test", folder=RQ1):
    z = np.load(folder / "preds" / f"{run}_{split}_decoded.npz", allow_pickle=False)
    preds = {s: z[f"pred_{s}"].astype(np.int8) for s in STRATEGIES}
    return z["sample_id"], z["true_label_vector"].astype(np.int8), z["probability_vector"], preds


# ---------------------------------------------------------------- bootstrap
def _f1_from_counts(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def _resampled_scores(W, y, pred, mask):
    """Macro-F1, Micro-F1 and Hamming loss for every bootstrap weight row of W."""
    y, pred = y[:, mask], pred[:, mask]
    tp = W @ (y & pred)
    fp = W @ ((1 - y) & pred)
    fn = W @ (y & (1 - pred))
    macro = _f1_from_counts(tp, fp, fn).mean(axis=1)
    micro = _f1_from_counts(tp.sum(1), fp.sum(1), fn.sum(1))
    hamming = (fp.sum(1) + fn.sum(1)) / (W.sum(1) * mask.sum())
    return {"macro_f1": macro, "micro_f1": micro, "hamming_loss": hamming}


def bootstrap(n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over test comments: every strategy is scored on the same resample."""
    _, tiers = _tiers()
    _, y, _, preds = load_decoded()
    n = len(y)
    rng = np.random.default_rng(seed)
    scopes = {"all": np.ones(len(LABELS), bool), **{t: tiers == t for t in ("head", "mid", "tail")}}
    observed_w = np.ones((1, n), dtype=np.int64)
    samples = {(s, scope): {m: [] for m in ("macro_f1", "micro_f1", "hamming_loss")}
               for s in STRATEGIES for scope in scopes}
    for start in range(0, n_boot, 1000):
        k = min(1000, n_boot - start)
        W = rng.multinomial(n, np.full(n, 1 / n), size=k).astype(np.int64)
        for s in STRATEGIES:
            for scope, mask in scopes.items():
                for metric, values in _resampled_scores(W, y, preds[s], mask).items():
                    samples[(s, scope)][metric].append(values)
    rows = []
    for a, b in COMPARISONS:
        for scope, mask in scopes.items():
            obs_a = _resampled_scores(observed_w, y, preds[a], mask)
            obs_b = _resampled_scores(observed_w, y, preds[b], mask)
            for metric in ("macro_f1", "micro_f1", "hamming_loss"):
                delta = np.concatenate(samples[(a, scope)][metric]) - np.concatenate(samples[(b, scope)][metric])
                better = delta < 0 if metric == "hamming_loss" else delta > 0
                rows.append({"comparison": f"{a} - {b}", "scope": scope, "metric": metric,
                             "observed_delta": float(obs_a[metric][0] - obs_b[metric][0]),
                             "ci95_low": float(np.percentile(delta, 2.5)),
                             "ci95_high": float(np.percentile(delta, 97.5)),
                             "share_resamples_a_better": float(better.mean()),
                             "n_boot": n_boot})
    out = pd.DataFrame(rows)
    out.to_csv(RQ1 / "tables" / "bootstrap_deltas.csv", index=False)
    return out


# ---------------------------------------------------------------- extra seeds
def seeds():
    """Repeat the frozen RQ1 protocol on two more PhoBERT training seeds (same config)."""
    from .analyze_thresholds import calibrate, evaluate

    rows, thresholds = [], {}
    runs = [(SEED, MAIN_RUN, RQ1)]
    for s in EXTRA_SEEDS:
        run = f"{MAIN_RUN}_seed{s}"
        if not (ROOT / "results" / "preds" / f"{run}_test.npz").exists():
            raise FileNotFoundError(f"Train {run} first: python -m src.train_phobert --emoji text --loss pw_sqrt "
                                    f"--epochs 12 --patience 3 --seed {s} --name {run}")
        folder = RQ1 / "seeds" / run
        calibrate(output_dir=folder, run=run)          # validation only; written before any test file is read
        evaluate(output_dir=folder, write_report=False)
        runs.append((s, run, folder))
    for s, run, folder in runs:
        spec = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        thresholds[s] = spec["per_label_thresholds"]
        summary = pd.read_csv(folder / "tables" / "strategy_metrics.csv")
        tiers = pd.read_csv(folder / "tables" / "tier_metrics.csv")
        gaps = pd.read_csv(folder / "tables" / "generalization_gaps.csv").set_index("strategy")
        for strategy in STRATEGIES:
            t = summary[(summary.split == "test") & (summary.strategy == strategy)].iloc[0]
            row = {"seed": s, "run": run, "strategy": strategy, "global_threshold": spec["global_threshold"],
                   "test_macro_f1": t.macro_f1, "test_micro_f1": t.micro_f1, "test_hamming_loss": t.hamming_loss,
                   "test_mean_predicted_cardinality": t.mean_predicted_cardinality,
                   "gap_val_minus_test_macro_f1": gaps.loc[strategy, "gap_val_minus_test_macro_f1"]}
            for tier in ("head", "mid", "tail"):
                row[f"test_macro_f1_{tier}"] = tiers[(tiers.split == "test") & (tiers.strategy == strategy)
                                                     & (tiers.tier == tier)].macro_f1.iloc[0]
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    # Does a label's validation gain from its own threshold transfer to test? (threshold overfitting check)
    transfer = []
    for s, run, folder in runs:
        comp = pd.read_csv(folder / "tables" / "per_label_comparison.csv")
        transfer.append({"seed": s, "run": run,
                         "spearman_val_gain_vs_test_gain": comp.val_gain_per_label_vs_fixed.corr(
                             comp.test_gain_per_label_vs_fixed, method="spearman"),
                         "labels_val_gain_positive": int((comp.val_gain_per_label_vs_fixed > 1e-12).sum()),
                         "labels_test_gain_positive": int((comp.test_gain_per_label_vs_fixed > 1e-12).sum()),
                         "labels_test_gain_negative": int((comp.test_gain_per_label_vs_fixed < -1e-12).sum()),
                         "mean_val_gain": comp.val_gain_per_label_vs_fixed.mean(),
                         "mean_test_gain": comp.test_gain_per_label_vs_fixed.mean()})
    pd.DataFrame(transfer).to_csv(RQ1 / "tables" / "seed_gain_transfer.csv", index=False)
    per_seed.to_csv(RQ1 / "tables" / "seed_results.csv", index=False)
    metrics = [c for c in per_seed.columns if c.startswith(("test_", "gap_"))]
    agg = per_seed.groupby("strategy")[metrics].agg(["mean", "std"])  # std with ddof=1 over 3 seeds
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.loc[list(STRATEGIES)].reset_index()
    agg.insert(1, "n_seeds", len(runs))
    agg.to_csv(RQ1 / "tables" / "seed_mean_std.csv", index=False)
    # Paired deltas per seed: the same backbone decoded three ways.
    piv = per_seed.pivot(index="seed", columns="strategy", values="test_macro_f1")
    deltas = pd.DataFrame({f"{a} - {b}": piv[a] - piv[b] for a, b in COMPARISONS})
    deltas.loc["mean"] = deltas.mean()
    deltas.loc["std"] = deltas.iloc[:-1].std()
    deltas.to_csv(RQ1 / "tables" / "seed_macro_f1_deltas.csv")
    _, tiers_arr = _tiers()
    th = pd.DataFrame(thresholds, index=LABELS)
    th.columns = [f"seed{c}" for c in th.columns]
    stab = th.copy()
    stab.insert(0, "tier", tiers_arr)
    stab["mean"], stab["std"] = th.mean(axis=1), th.std(axis=1)
    stab["range"] = th.max(axis=1) - th.min(axis=1)
    stab.index.name = "label"
    stab.reset_index().to_csv(RQ1 / "tables" / "seed_threshold_stability.csv", index=False)
    _plot_seeds(per_seed, stab)
    return per_seed, agg, deltas, stab


def _plot_seeds(per_seed, stab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    for i, strategy in enumerate(STRATEGIES):
        v = per_seed[per_seed.strategy == strategy].test_macro_f1
        ax.scatter(np.full(len(v), i), v, color="black", s=18, zorder=3)
        ax.errorbar(i, v.mean(), yerr=v.std(), fmt="s", color="gray", capsize=6, zorder=2)
    ax.set_xticks(range(3), STRATEGIES)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Three training seeds (dots), mean ± std")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    d = stab.sort_values("mean")
    seed_cols = [c for c in stab.columns if c.startswith("seed")]
    for c, marker in zip(seed_cols, ("o", "^", "s")):
        ax.scatter(d[c], range(len(d)), marker=marker, s=16, facecolors="none", edgecolors="black", label=c)
    ax.hlines(range(len(d)), d[seed_cols].min(axis=1), d[seed_cols].max(axis=1), color="gray", lw=1)
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_yticks(range(len(d)), d.index, fontsize=7)
    ax.set_xlabel("Per-label threshold selected on validation")
    ax.set_title("Per-label threshold across seeds")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RQ1 / "figures" / "seed_robustness.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- error audit
def _names(row):
    return "; ".join(LABELS[j] for j in np.flatnonzero(row))


def _phi_train():
    Y = label_matrix(load_split("train")).astype(float)
    return np.corrcoef(Y, rowvar=False)


def errors(seed=SEED):
    spec, tiers = _tiers()
    ids, y, p, preds = load_decoded()
    out = RQ1 / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    th = np.asarray(spec["per_label_thresholds"])
    # 1. Per-label error counts for each decoder.
    rows = []
    for s in STRATEGIES:
        pr = preds[s]
        for k, label in enumerate(LABELS):
            rows.append({"strategy": s, "label": label, "tier": tiers[k],
                         "tp": int((y[:, k] & pr[:, k]).sum()), "fp": int(((1 - y[:, k]) & pr[:, k]).sum()),
                         "fn": int((y[:, k] & (1 - pr[:, k])).sum()), "support": int(y[:, k].sum())})
    counts = pd.DataFrame(rows)
    counts.to_csv(out / "error_counts.csv", index=False)
    # 2. Decisions that change between decoders (per label).
    flips = []
    for a, b in (("per_label", "fixed"), ("per_label", "global"), ("global", "fixed")):
        pa, pb = preds[a], preds[b]
        for k, label in enumerate(LABELS):
            yk = y[:, k] == 1
            flips.append({"comparison": f"{a} vs {b}", "label": label, "tier": tiers[k],
                          "threshold_a": float(th[k]) if a == "per_label" else
                          (spec["global_threshold"] if a == "global" else 0.5),
                          "threshold_b": 0.5 if b == "fixed" else spec["global_threshold"],
                          "fn_recovered": int((yk & (pa[:, k] == 1) & (pb[:, k] == 0)).sum()),
                          "fn_introduced": int((yk & (pa[:, k] == 0) & (pb[:, k] == 1)).sum()),
                          "fp_removed": int((~yk & (pa[:, k] == 0) & (pb[:, k] == 1)).sum()),
                          "fp_introduced": int((~yk & (pa[:, k] == 1) & (pb[:, k] == 0)).sum())})
    flips = pd.DataFrame(flips)
    flips["net_correct_decisions"] = (flips.fn_recovered + flips.fp_removed
                                      - flips.fn_introduced - flips.fp_introduced)
    flips.to_csv(out / "decision_flips.csv", index=False)
    # 3. Confused pairs: true label i missed while label j predicted spuriously in the same comment;
    #    and spurious companions: label i correct, label j added without gold support.
    phi = _phi_train()
    pair_rows = []
    for s in STRATEGIES:
        pr = preds[s]
        fn = (y == 1) & (pr == 0)
        fp = (y == 0) & (pr == 1)
        tp = (y == 1) & (pr == 1)
        confused = fn.T.astype(int) @ fp.astype(int)
        companion = tp.T.astype(int) @ fp.astype(int)
        for kind, M in (("missed_i_predicted_j", confused), ("correct_i_spurious_j", companion)):
            for i in range(len(LABELS)):
                for j in range(len(LABELS)):
                    if i != j and M[i, j] > 0:
                        pair_rows.append({"strategy": s, "kind": kind, "label_i": LABELS[i], "label_j": LABELS[j],
                                          "n_comments": int(M[i, j]), "phi_train": round(float(phi[i, j]), 3)})
    pairs = pd.DataFrame(pair_rows).sort_values(["strategy", "kind", "n_comments"], ascending=[True, True, False])
    pairs.to_csv(out / "confused_pairs.csv", index=False)
    # 4. Cardinality: under/over-prediction by number of gold labels.
    card = y.sum(1)
    group = np.where(card >= 4, ">=4", card.astype(str))
    crow = []
    for s in STRATEGIES:
        pc = preds[s].sum(1)
        exact = (preds[s] == y).all(1)
        for g in ("1", "2", "3", ">=4"):
            m = group == g
            crow.append({"strategy": s, "cardinality": g, "n_samples": int(m.sum()),
                         "mean_true": float(card[m].mean()), "mean_pred": float(pc[m].mean()),
                         "share_under": float((pc[m] < card[m]).mean()), "share_equal_count": float((pc[m] == card[m]).mean()),
                         "share_over": float((pc[m] > card[m]).mean()), "share_empty": float((pc[m] == 0).mean()),
                         "share_exact_set": float(exact[m].mean())})
    pd.DataFrame(crow).to_csv(out / "cardinality_errors.csv", index=False)
    # 5. Case sample for the qualitative audit, drawn before any text is read.
    cases = _sample_cases(ids, y, p, preds, tiers, th, seed)
    cases.to_csv(out / "cases.csv", index=False)
    return counts, flips, pairs, cases


def _sample_cases(ids, y, p, preds, tiers, th, seed):
    rng = np.random.default_rng(seed)
    midterm = set(pd.read_csv(ROOT / "results" / "error_cases" / "cases.csv", dtype=str).sample_id)
    fresh = np.array([i not in midterm for i in ids])  # do not reread the 70 Midterm cases
    pl, fx = preds["per_label"], preds["fixed"]
    used = set()

    def pick(mask, n, group, per_tier=None):
        """Sample (comment, label) cells from mask, at most one per comment across all groups."""
        rows = []
        cells = np.argwhere(mask & fresh[:, None])
        rng.shuffle(cells)
        quota = dict(per_tier) if per_tier else None
        for i, k in cells:
            if len(rows) == n:
                break
            if ids[i] in used or (quota is not None and quota.get(tiers[k], 0) == 0):
                continue
            if quota is not None:
                quota[tiers[k]] -= 1
            used.add(ids[i])
            rows.append((group, i, k))
        if len(rows) < n:
            raise RuntimeError(f"{group}: only {len(rows)} cases available")
        return rows

    strat = {"head": 7, "mid": 7, "tail": 6}
    sel = []
    sel += pick((y == 1) & (pl == 0), 20, "fn_per_label", strat)            # misses that remain after tuning
    sel += pick((y == 0) & (pl == 1), 20, "fp_per_label", strat)            # spurious labels after tuning
    sel += pick((y == 1) & (pl == 0) & (fx == 1), 10, "fn_introduced")      # cost of raising a threshold
    sel += pick((y == 0) & (pl == 1) & (fx == 0), 10, "fp_introduced")      # cost of lowering a threshold
    sel += pick((y == 1) & (pl == 1) & (fx == 0), 5, "fn_recovered")        # gain of lowering a threshold
    sel += pick((y == 0) & (pl == 0) & (fx == 1), 5, "fp_removed")          # gain of raising a threshold
    multi = np.flatnonzero((y.sum(1) >= 4) & fresh)
    multi = [i for i in rng.permutation(multi) if ids[i] not in used][:10]
    rows = []
    for group, i, k in sel:
        rows.append({"group": group, "sample_id": ids[i], "label": LABELS[k], "tier": tiers[k],
                     "prob": round(float(p[i, k]), 4), "threshold_per_label": float(th[k]),
                     "true_labels": _names(y[i]), "pred_fixed": _names(fx[i]),
                     "pred_global": _names(preds["global"][i]), "pred_per_label": _names(pl[i]),
                     "n_true": int(y[i].sum())})
    for i in multi:
        rows.append({"group": "multi_label", "sample_id": ids[i], "label": "", "tier": "", "prob": np.nan,
                     "threshold_per_label": np.nan, "true_labels": _names(y[i]), "pred_fixed": _names(fx[i]),
                     "pred_global": _names(preds["global"][i]), "pred_per_label": _names(pl[i]),
                     "n_true": int(y[i].sum())})
    return pd.DataFrame(rows)


CAUSES = {"near_synonym": "Near-synonymous labels", "keyword": "Lexical cue / keyword trap",
          "sarcasm": "Sarcasm or dark humour", "emoji": "Emoji or emoticon cue",
          "context": "Missing context / very short", "noisy": "Teencode, dialect or noisy text",
          "mixed": "Mixed or secondary emotions", "gold": "Questionable or incomplete gold label",
          "borderline": "Borderline score near 0.5", "threshold": "Decision flipped by the tuned threshold"}


def summarize_annotations():
    """Tally the manual audit (results/rq1/error_analysis/annotations.csv) and draw the error figures."""
    out = RQ1 / "error_analysis"
    ann = pd.read_csv(out / "annotations.csv", dtype=str, keep_default_na=False)
    cases = pd.read_csv(out / "cases.csv", dtype=str, keep_default_na=False)
    key = ["group", "sample_id", "label"]
    if not ann[key].equals(cases[key]):
        raise ValueError("annotations.csv must list exactly the sampled cases, in the same order")
    unknown = set(ann.cause_primary) - set(CAUSES) | (set(ann.cause_secondary) - set(CAUSES) - {""})
    if unknown:
        raise ValueError(f"Unknown cause codes: {unknown}")
    tally = pd.crosstab(ann.cause_primary.map(CAUSES), ann.group, margins=True, margins_name="total")
    order = ["fn_per_label", "fp_per_label", "fn_introduced", "fp_introduced", "fn_recovered", "fp_removed",
             "multi_label", "total"]
    tally = tally[[c for c in order if c in tally.columns]]
    body = tally.drop(index="total").sort_values("total", ascending=False)
    tally = pd.concat([body, tally.loc[["total"]]])
    tally.to_csv(out / "cause_summary.csv")
    any_mention = pd.concat([ann.cause_primary, ann.cause_secondary[ann.cause_secondary != ""]]).map(CAUSES)
    any_mention.value_counts().rename("primary_or_secondary").to_csv(out / "cause_mentions.csv")
    defens = pd.crosstab(ann.group, ann.prediction_defensible)
    defens.to_csv(out / "defensible_by_group.csv")
    _plot_errors(out)
    return tally, defens


def _plot_errors(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flips = pd.read_csv(out / "decision_flips.csv")
    d = flips[flips.comparison == "per_label vs fixed"].sort_values("threshold_a")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    y = np.arange(len(d))
    gains = d.fn_recovered + d.fp_removed
    losses = d.fn_introduced + d.fp_introduced
    ax.barh(y, gains, color="#9e9e9e", label="decisions corrected (FN recovered + FP removed)")
    ax.barh(y, -losses, color="#404040", label="decisions broken (FN introduced + FP introduced)")
    ax.scatter(gains - losses, y, color="white", edgecolors="black", zorder=3, s=22, label="net")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [f"{l} ({t:.3f})" for l, t in zip(d.label, d.threshold_a)], fontsize=8)
    ax.set_xlabel("Test decisions changed: per-label vs fixed 0.5 (label, tuned threshold)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RQ1 / "figures" / "decision_flips.png", dpi=160)
    plt.close(fig)
    c = pd.read_csv(out / "cardinality_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = ["1", "2", "3", ">=4"]
    width = 0.27
    for i, (s, hatch) in enumerate(zip(STRATEGIES, ("", "//", ".."))):
        sub = c[c.strategy == s].set_index("cardinality").loc[groups]
        x = np.arange(4) + (i - 1) * width
        axes[0].bar(x, sub.share_under, width, color="white", edgecolor="black", hatch=hatch, label=s)
        axes[1].bar(x, sub.share_over, width, color="white", edgecolor="black", hatch=hatch, label=s)
    for ax, title in zip(axes, ("Under-predicted (fewer labels than gold)", "Over-predicted (more labels than gold)")):
        ax.set_xticks(range(4), ["|Y|=1", "|Y|=2", "|Y|=3", "|Y|≥4"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of test comments")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RQ1 / "figures" / "cardinality_errors.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("bootstrap", "seeds", "errors", "summary", "all"), default="all")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.part in ("bootstrap", "all"):
        b = bootstrap()
        print(b[b.scope == "all"].round(4).to_string(index=False))
    if a.part in ("errors", "all"):
        _, flips, _, cases = errors()
        print(flips.groupby("comparison")[["fn_recovered", "fn_introduced", "fp_removed", "fp_introduced",
                                           "net_correct_decisions"]].sum())
        print(cases.group.value_counts().to_string())
    if a.part in ("summary", "all"):
        tally, defens = summarize_annotations()
        print(tally.to_string())
        print(defens.to_string())
    if a.part in ("seeds", "all"):
        _, agg, deltas, stab = seeds()
        print(agg.round(4).to_string(index=False))
        print(deltas.round(4).to_string())
        print("mean per-label threshold range across seeds:", round(float(stab["range"].mean()), 3))


if __name__ == "__main__":
    main()
