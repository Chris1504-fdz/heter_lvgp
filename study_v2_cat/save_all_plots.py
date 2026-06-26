#!/usr/bin/env python
"""
Regenerate the whole plots/ tree for study_v2_gp from the SAME StudyResults class the
notebook uses (reused from study_v2, so saved figures == inline figures), plus the
per-category-GP vs LVGP comparison overlays.

    plots/main/         overview, convergence, x1, input-space, best-design plots
    plots/analysis/     analytical gallery (regret, pareto, n_rep, heatmaps, runtime) + tables
    plots/single_runs/  one per-category slice plot per acquisition (n_rep=10, seed 1)
    plots/comparison/   per-category GP vs LVGP head-to-head overlays  (Part-2 headline)

Usage:  python save_all_plots.py                 # all sections + comparison
        python save_all_plots.py --single        # only single_runs
        python save_all_plots.py --no-compare    # skip the LVGP overlays
        python save_all_plots.py --lvgp PATH      # LVGP results dir (default ../study_v2/results)
Re-runnable any time; reads results/, never re-runs the optimization.
"""
import os, sys, argparse
import matplotlib   # NB: Agg is set only under __main__ (below), so importing this module
                    # into a notebook to call main() does NOT override the inline backend

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from utils import StudyResults, compare_studies, compare_studies_multi   # noqa: E402
from utils.problem import CONFIG_ORDER, acf_tag           # noqa: E402

PLOTS = os.path.join(HERE, "plots")
MAIN = os.path.join(PLOTS, "main")
ANALYSIS = os.path.join(PLOTS, "analysis")
SINGLE = os.path.join(PLOTS, "single_runs")
COMPARE = os.path.join(PLOTS, "comparison")
for d in (MAIN, ANALYSIS, SINGLE, COMPARE):
    os.makedirs(d, exist_ok=True)


def _save(fig, folder, name):
    p = os.path.join(folder, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  saved", os.path.relpath(p, HERE))


def save_main(study):
    print("plots/main:")
    _save(study.plot_progress(), MAIN, "0_progress.png")
    _save(study.plot_initial_doe(), MAIN, "0b_initial_doe.png")
    _save(study.plot_convergence(), MAIN, "1_convergence_noisy.png")
    _save(study.plot_convergence_true(log=True), MAIN, "1b_convergence_true_log.png")
    _save(study.plot_convergence_true(ymax=4), MAIN, "1b_convergence_true_zoom.png")
    _save(study.plot_x1_convergence(is_ground_truth=False), MAIN, "1c_x1_convergence_noisy.png")
    _save(study.plot_x1_convergence(is_ground_truth=True), MAIN, "1c_x1_convergence_true.png")
    _save(study.plot_x1_distribution(n_rep=10, show_violin=False, show_mean=False, is_ground_truth=False),
          MAIN, "1d_x1_distribution_noisy.png")
    _save(study.plot_x1_distribution(n_rep=10, show_violin=False, show_mean=False, is_ground_truth=True),
          MAIN, "1d_x1_distribution_true.png")
    _save(study.plot_x1_landing(n_rep=10), MAIN, "1e_x1_landing.png")
    _save(study.plot_input_space(n_rep=10), MAIN, "2_input_space.png")
    _save(study.plot_best_designs(n_rep=10, is_ground_truth=False), MAIN, "2b_best_designs_noisy.png")
    _save(study.plot_best_designs(n_rep=10, is_ground_truth=True), MAIN, "2b_best_designs_true.png")
    _save(study.plot_level_histogram(), MAIN, "3_level_histogram.png")
    _save(study.plot_objective_variance(is_ground_truth=False), MAIN, "4_objective_variance_noisy.png")
    _save(study.plot_objective_variance(is_ground_truth=True), MAIN, "4_objective_variance_true.png")
    _save(study.plot_final_boxplots(is_ground_truth=False), MAIN, "5_final_boxplots_noisy.png")
    _save(study.plot_final_boxplots(is_ground_truth=True), MAIN, "5_final_boxplots_true.png")
    _save(study.plot_best_per_category(n_rep=10, is_ground_truth=False), MAIN, "6_best_per_category_noisy.png")
    _save(study.plot_best_per_category(n_rep=10, is_ground_truth=True), MAIN, "6_best_per_category_true.png")


def save_analysis(study):
    print("plots/analysis:")
    _save(study.plot_simple_regret(is_ground_truth=False), ANALYSIS, "A_simple_regret_noisy.png")
    _save(study.plot_simple_regret(is_ground_truth=True), ANALYSIS, "A_simple_regret_true.png")
    _save(study.plot_pareto(is_ground_truth=False), ANALYSIS, "B_pareto_obj_var_noisy.png")
    _save(study.plot_pareto(is_ground_truth=True), ANALYSIS, "B_pareto_obj_var_true.png")
    _save(study.plot_incumbent_variance(is_ground_truth=False), ANALYSIS, "C_incumbent_variance_noisy.png")
    _save(study.plot_incumbent_variance(is_ground_truth=True), ANALYSIS, "C_incumbent_variance_true.png")
    _save(study.plot_nrep_sensitivity(is_ground_truth=False), ANALYSIS, "D_nrep_sensitivity_noisy.png")
    _save(study.plot_nrep_sensitivity(is_ground_truth=True), ANALYSIS, "D_nrep_sensitivity_true.png")
    _save(study.plot_summary_heatmaps(is_ground_truth=False), ANALYSIS, "E_summary_heatmaps_noisy.png")
    _save(study.plot_summary_heatmaps(is_ground_truth=True), ANALYSIS, "E_summary_heatmaps_true.png")
    _save(study.plot_runtime(), ANALYSIS, "F_runtime.png")
    study.regret_table(is_ground_truth=True, eps=0.1,
                       excel_path=os.path.join(ANALYSIS, "simple_regret_table.xlsx"))
    study.incumbent_variance_table(is_ground_truth=True, tol=0.15,
                       excel_path=os.path.join(ANALYSIS, "incumbent_variance_table.xlsx"))


def save_single_runs(study):
    print("plots/single_runs (n_rep=10, seed 1):")
    for acf, p in CONFIG_ORDER:
        rel = f"{acf_tag(acf, p)}/nrep10/seed01.npz"       # .npz (this study)
        if not os.path.exists(os.path.join(study.results_dir, rel)):
            print("  skip (missing)", rel); continue
        plabel = "" if p != p else f"_{p:g}"
        _save(study.plot_single_run(rel), SINGLE, f"single_run_{acf}{plabel}_nr10_s1.png")


def save_comparison(study, others):
    """Overlay this categorical-GP study against `others` (list of (StudyResults, label)).
    Produces pairwise figures vs each, plus a single N-way figure with all studies."""
    this_label = "Categorical GP"
    print("plots/comparison (categorical GP vs others):")
    for s, lab in others:                                    # pairwise vs each study
        tag = lab.lower().replace(" ", "").replace("-", "")
        for metric in ("true_best_sampled", "best_y"):       # noiseless (headline) + noisy
            for nr in (3, 5, 10):
                try:
                    fig = compare_studies(study, s, metric=metric, n_rep=nr,
                                          labels=(this_label, lab))
                except ValueError as e:
                    print(f"  skip {lab} {metric} nrep{nr:02d}: {e}"); continue
                _save(fig, COMPARE, f"compare_{tag}_{metric}_nrep{nr:02d}.png")
    series = list(others) + [(study, this_label)]            # e.g. LVGP, Per-category, Categorical
    if len(series) >= 3:                                      # the 3-way headline
        for metric in ("true_best_sampled", "best_y"):       # noiseless (headline) + noisy
            for nr in (3, 5, 10):
                try:
                    fig = compare_studies_multi(series, metric=metric, n_rep=nr)
                except ValueError as e:
                    print(f"  skip 3-way {metric} nrep{nr:02d}: {e}"); continue
                _save(fig, COMPARE, f"compare_ALL_{metric}_nrep{nr:02d}.png")


def main(results_dir="results", lvgp_dir="../study_v2/results",
         percat_dir="../study_v2_gp/results", single_only=False, compare=True):
    study = StudyResults.load(os.path.join(HERE, results_dir))
    if not study.runs:
        print("no results yet — run the sweep first."); return
    print(f"loaded {len(study.runs)} runs")
    if not single_only:
        save_main(study)
        save_analysis(study)
    save_single_runs(study)
    if compare and not single_only:
        others = []
        for d, lab in [(lvgp_dir, "LVGP"), (percat_dir, "Per-category GP")]:
            path = d if os.path.isabs(d) else os.path.join(HERE, d)
            if os.path.isdir(path):
                s = StudyResults.load(path)
                if s.runs:
                    others.append((s, lab))
                else:
                    print(f"(no runs under {path}; skipping {lab})")
            else:
                print(f"({lab} dir {path} not found; skipping)")
        if others:
            save_comparison(study, others)
    print("done.")


if __name__ == "__main__":
    matplotlib.use("Agg")                # headless-safe for command-line use
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="only single_runs")
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison overlays")
    ap.add_argument("--lvgp", default="../study_v2/results", help="LVGP results dir")
    ap.add_argument("--percat-gp", default="../study_v2_gp/results", help="per-category GP results dir")
    args = ap.parse_args()
    main(lvgp_dir=args.lvgp, percat_dir=args.percat_gp,
         single_only=args.single, compare=not args.no_compare)
