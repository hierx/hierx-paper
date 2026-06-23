#!/usr/bin/env python3
"""Plot Figure 7: error and bias anatomy of the GB case study.

Three panels from error_anatomy_gb results:
(a) relative distance error vs. exact OD distance (median, p5-p95 band,
    and signed bias),
(b) signed relative interaction error vs. distance for kernels of
    decreasing steepness (bias per distance bin),
(c) accessibility error vs. activity support size: sparse supports
    (exact ground truth on the full network) compared with the ubiquitous
    census population vector (exact ground truth at sampled nodes).

Usage:
    python -m benchmarks_final.plot_error_anatomy [--quick]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks_final.common import DATA_DIR, FIGURES_DIR, STYLE, TABLES_DIR

KERNEL_COLORS = {
    "very_steep": "#b2182b",
    "steep_paper": "#ef8a62",
    "moderate": "#67a9cf",
    "shallow": "#2166ac",
}

# Distinct marker + line style per kernel so panel (b) is legible in
# grayscale, not only by color (consistent with the other paper figures).
KERNEL_MARKERS = {
    "very_steep": "o",
    "steep_paper": "s",
    "moderate": "^",
    "shallow": "D",
}
KERNEL_LINESTYLES = {
    "very_steep": "-",
    "steep_paper": "--",
    "moderate": "-.",
    "shallow": ":",
}


def norm_stats(pairs_npz, act: str, k_name: str) -> dict:
    """Norm-weighted error stats at sampled origins from the raw NPZ."""
    he = pairs_npz[f"h_exact__{act}__{k_name}"]
    hh = pairs_npz[f"h_hier_at_origins__{act}__{k_name}"]
    mass = float(hh.sum() / he.sum())
    return {
        "norm_rmse": float(np.linalg.norm(hh - he) / np.linalg.norm(he)),
        "mass_ratio": mass,
        "calibrated_rmse": float(
            np.linalg.norm(hh / mass - he) / np.linalg.norm(he)
        ),
    }


def jensen_prediction(results: dict, pairs_npz) -> dict:
    """Second-order (Jensen) prediction of the systematic interaction bias.

    The far-field relative distance error e = (c_hier - c_exact)/c_exact has
    mean mu and std sigma (computed over pairs beyond the finest-layer cutoff
    alpha*rho0, where grouping first occurs). For a convex decreasing kernel
    f(c) = (c+delta0)^{-beta}, a second-order expansion gives the expected
    relative interaction error  ~= -beta*mu + 0.5*beta*(beta+1)*sigma^2.

    Returns mu, sigma, the cutoff used, and per-kernel predicted vs observed
    accessibility bias. This makes the manuscript's Jensen numbers
    reproducible from the committed pipeline rather than an ad-hoc script.
    """
    meta = results["metadata"]
    de = pairs_npz["pair_d_exact"]
    ch = pairs_npz["pair_c_hier"]
    cutoff = meta["overlap_factor"] * meta["base_radius"]
    mask = np.isfinite(ch) & (de > cutoff)
    e = (ch[mask] - de[mask]) / de[mask]
    mu, sigma = float(e.mean()), float(e.std())

    # power-law exponent beta parsed from each kernel label "(c+d)^{-b}"
    betas = {"steep_paper": 2.0, "very_steep": 2.0, "moderate": 1.5,
             "shallow": 1.0}
    out = {"cutoff_s": cutoff, "n_far_field_pairs": int(mask.sum()),
           "mu": mu, "sigma": sigma, "per_kernel": {}}
    acc = results["accessibility_errors_at_origins"]
    for k_name, beta in betas.items():
        if k_name not in meta["kernels"]:
            continue
        pred = -beta * mu + 0.5 * beta * (beta + 1.0) * sigma**2
        obs = acc.get(f"population__{k_name}", {}).get("bias_rel")
        out["per_kernel"][k_name] = {
            "beta": beta,
            "predicted_bias": pred,
            "observed_bias": obs,
        }
    return out


def write_tables(results: dict, pairs_npz, suffix: str) -> None:
    """Emit SI tables for accessibility errors by kernel and by sparsity."""
    meta = results["metadata"]
    acc = results["accessibility_errors_at_origins"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accessibility error of the hierarchical operator at "
        f"{meta['n_origins']} sampled nodes of the GB driving network "
        r"($n = 2{,}576{,}491$), against exact Dijkstra ground truth. "
        r"Bias, p5, and p95 are per-node signed relative errors; positive "
        r"bias means the operator overestimates accessibility. "
        r"$\varepsilon$ is the norm-weighted relative RMSE "
        r"$\|\mathbf{h}^{\mathrm{hier}} - \mathbf{h}^{\mathrm{exact}}\| / "
        r"\|\mathbf{h}^{\mathrm{exact}}\|$ over the sample; "
        r"$\varepsilon_{\mathrm{cal}}$ is the same after dividing by the "
        r"mass ratio $\sum h^{\mathrm{hier}} / \sum h^{\mathrm{exact}}$.}",
        r"\label{tab:si-error-anatomy-kernels}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"activity & kernel & bias & p5 & p95 & $\varepsilon$ & "
        r"mass & $\varepsilon_{\mathrm{cal}}$ \\",
        r"\midrule",
    ]
    for key, st in acc.items():
        act, k_name = key.split("__")
        label = f"${meta['kernels'][k_name]}$"
        ns = norm_stats(pairs_npz, act, k_name)
        lines.append(
            f"{act} & {label} & {st['bias_rel']*100:+.1f}\\% & "
            f"{st['p5']*100:+.1f}\\% & {st['p95']*100:+.1f}\\% & "
            f"{ns['norm_rmse']*100:.1f}\\% & {ns['mass_ratio']:.3f} & "
            f"{ns['calibrated_rmse']*100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = TABLES_DIR / f"error_anatomy_kernels{suffix}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved {out}")

    sparse = results["sparse_activity_results"]
    sizes = sorted({r["support_size"] for r in sparse})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of activity sparsity on accessibility error "
        r"(GB driving network, kernel "
        f"${meta['kernels'][meta['paper_kernel']]}$). "
        r"For each support size $k$, unit activity is placed on $k$ "
        r"population-weighted sample nodes; exact ground truth for the "
        r"full network is computed with $k$ Dijkstra runs. Mean over "
        f"{meta['n_trials']} trials (RMSE range in brackets).}}",
        r"\label{tab:si-error-anatomy-sparse}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$k$ & RMSE & bias & p95 $|$rel.\ err$|$ \\",
        r"\midrule",
    ]
    for k in sizes:
        rows = [r for r in sparse if r["support_size"] == k]
        rmse = [r["relative_rmse"] for r in rows]
        bias = [r["bias_rel"] for r in rows]
        p95 = [max(abs(r["p5"]), abs(r["p95"])) for r in rows]
        lines.append(
            f"{k} & {np.mean(rmse)*100:.1f}\\% "
            f"[{min(rmse)*100:.1f}--{max(rmse)*100:.1f}] & "
            f"{np.mean(bias)*100:+.1f}\\% & {np.mean(p95)*100:.1f}\\% \\\\"
        )
    pop_key = f"population__{meta['paper_kernel']}"
    if pop_key in acc:
        st = acc[pop_key]
        ns = norm_stats(pairs_npz, "population", meta["paper_kernel"])
        lines.append(r"\midrule")
        lines.append(
            f"census pop.\\ & {ns['norm_rmse']*100:.1f}\\%$^{{\\dagger}}$ & "
            f"{st['bias_rel']*100:+.1f}\\% & "
            f"{max(abs(st['p5']), abs(st['p95']))*100:.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\smallskip",
        r"\raggedright\footnotesize $^{\dagger}$\,Census-population row "
        f"evaluated at {meta['n_origins']} sampled nodes (exact ground "
        r"truth per node); sparse-support rows evaluated over all "
        r"reachable nodes.",
        r"\end{table}",
    ]
    out = TABLES_DIR / f"error_anatomy_sparse{suffix}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use the _quick (London smoke test) results")
    args = parser.parse_args()

    suffix = "_quick" if args.quick else ""
    with open(DATA_DIR / f"error_anatomy_gb_results{suffix}.json") as f:
        results = json.load(f)
    pairs_npz = np.load(DATA_DIR / f"error_anatomy_gb_pairs{suffix}.npz")

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    bins = results["pair_stats_by_distance_bin"]
    # Drop sparsely populated bins (unstable percentiles)
    bins = [b for b in bins if b["n_pairs"] >= 50]
    x = np.array([np.sqrt(b["bin_lo"] * b["bin_hi"]) if b["bin_lo"] > 0
                  else b["bin_hi"] / 2 for b in bins]) / 60.0  # minutes

    # ---- Panel (a): distance error vs distance ----
    ax = axes[0]
    p5 = np.array([b["dist_err_rel_p5"] for b in bins]) * 100
    p50 = np.array([b["dist_err_rel_p50"] for b in bins]) * 100
    p95 = np.array([b["dist_err_rel_p95"] for b in bins]) * 100
    bias = np.array([b["dist_bias_rel"] for b in bins]) * 100
    ax.fill_between(x, p5, p95, alpha=0.25, color="#67a9cf",
                    label="5th--95th percentile")
    ax.plot(x, p50, "-", color="#2166ac", label="median")
    ax.plot(x, bias, "--", color="#b2182b", label="mean (bias)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Exact OD travel time (min)")
    ax.set_ylabel("Relative distance error (%)")
    ax.set_title("(a) Distance error", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    # ---- Panel (b): signed interaction error vs distance, by kernel ----
    ax = axes[1]
    kernel_labels = results["metadata"]["kernels"]
    for k_name in ["very_steep", "steep_paper", "moderate", "shallow"]:
        if k_name not in kernel_labels:
            continue
        kb = np.array([b["kernels"][k_name]["bias_rel"] for b in bins]) * 100
        lbl = f"${kernel_labels[k_name]}$"
        if k_name == results["metadata"]["paper_kernel"]:
            lbl += " (map kernel)"
        ax.plot(x, kb, linestyle=KERNEL_LINESTYLES[k_name],
                marker=KERNEL_MARKERS[k_name], ms=4, markevery=2,
                color=KERNEL_COLORS[k_name], label=lbl)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Exact OD travel time (min)")
    ax.set_ylabel("Mean signed interaction error (%)")
    ax.set_title("(b) Interaction bias by kernel", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    # ---- Panel (c): error vs activity support size ----
    ax = axes[2]
    sparse = results["sparse_activity_results"]
    sizes = sorted({r["support_size"] for r in sparse})
    rmse_mean, rmse_lo, rmse_hi, bias_mean = [], [], [], []
    for k in sizes:
        vals = [r["relative_rmse"] for r in sparse if r["support_size"] == k]
        bvals = [r["bias_rel"] for r in sparse if r["support_size"] == k]
        rmse_mean.append(np.mean(vals) * 100)
        rmse_lo.append(np.min(vals) * 100)
        rmse_hi.append(np.max(vals) * 100)
        bias_mean.append(np.mean(bvals) * 100)
    rmse_mean = np.array(rmse_mean)
    yerr = np.array([rmse_mean - rmse_lo, np.array(rmse_hi) - rmse_mean])
    ax.errorbar(sizes, rmse_mean, yerr=yerr, fmt="o-", color="#b2182b",
                capsize=3, label="sparse support, RMSE")
    ax.plot(sizes, bias_mean, "s--", ms=4, color="#ef8a62",
            label="sparse support, bias")

    acc = results["accessibility_errors_at_origins"]
    pop_key = f"population__{results['metadata']['paper_kernel']}"
    if pop_key in acc:
        ns = norm_stats(pairs_npz, "population",
                        results["metadata"]["paper_kernel"])
        ax.axhline(ns["norm_rmse"] * 100, color="#2166ac", ls="-",
                   lw=1.2, label="census population, RMSE")
        ax.axhline(acc[pop_key]["bias_rel"] * 100, color="#67a9cf", ls="--",
                   lw=1.2, label="census population, bias")
    ax.set_xscale("log")
    ax.set_xlabel("Activity support size $k$ (nodes)")
    ax.set_ylabel("Accessibility error (%)")
    ax.set_title("(c) Effect of activity sparsity", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"paper_fig7_error_anatomy{suffix}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")

    write_tables(results, pairs_npz, suffix)

    jensen = jensen_prediction(results, pairs_npz)
    jpath = DATA_DIR / f"error_anatomy_jensen{suffix}.json"
    with open(jpath, "w") as f:
        json.dump(jensen, f, indent=2)
    print(f"Saved {jpath}")
    print(
        f"  Jensen: far-field (d>{jensen['cutoff_s']:.0f}s, "
        f"n={jensen['n_far_field_pairs']:,})  "
        f"mu={jensen['mu']:+.4f}  sigma={jensen['sigma']:.4f}"
    )
    for k_name, pk in jensen["per_kernel"].items():
        obs = pk["observed_bias"]
        obs_s = f"{obs*100:+.1f}%" if obs is not None else "n/a"
        print(
            f"    {k_name:12s} beta={pk['beta']}: "
            f"predicted {pk['predicted_bias']*100:+.1f}%  observed {obs_s}"
        )


if __name__ == "__main__":
    main()
