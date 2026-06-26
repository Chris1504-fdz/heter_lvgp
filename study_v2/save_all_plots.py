#!/usr/bin/env python
"""
Regenerate the whole dedicated plots/ tree for study_v2 from the same StudyResults
class the notebook uses, so saved figures == inline figures.

    plots/main/         overview, convergence, x1, input-space, best-design plots
    plots/analysis/     the analytical gallery (regret, pareto, n_rep, heatmaps, runtime)
    plots/single_runs/  one per-category slice plot per acquisition (n_rep=10, seed 1)
    plots/doe_preview/  (left untouched — the DOE design previews)

Usage:  python save_all_plots.py            # all sections
        python save_all_plots.py --single   # only single_runs
Re-runnable any time; reads results/, never re-runs the optimization.
"""
import os, sys, argparse
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from utils import StudyResults                          # noqa: E402
from utils.problem import CONFIG_ORDER, acf_tag         # noqa: E402

PLOTS = os.path.join(HERE, "plots")
MAIN = os.path.join(PLOTS, "main")
ANALYSIS = os.path.join(PLOTS, "analysis")
SINGLE = os.path.join(PLOTS, "single_runs")
for d in (MAIN, ANALYSIS, SINGLE):
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
    # convergence: noisy + true twin
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
    # objective-value plots: noisy + ground truth
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
    # tabular companions (styled .xlsx) to the simple-regret + incumbent-variance plots
    study.regret_table(is_ground_truth=True, eps=0.1,
                       excel_path=os.path.join(ANALYSIS, "simple_regret_table.xlsx"))
    study.incumbent_variance_table(is_ground_truth=True, tol=0.15,
                       excel_path=os.path.join(ANALYSIS, "incumbent_variance_table.xlsx"))


def save_single_runs(study):
    print("plots/single_runs (n_rep=10, seed 1):")
    for acf, p in CONFIG_ORDER:
        rel = f"{acf_tag(acf, p)}/nrep10/seed01.mat"
        if not os.path.exists(os.path.join(study.results_dir, rel)):
            print("  skip (missing)", rel); continue
        plabel = "" if p != p else f"_{p:g}"
        _save(study.plot_single_run(rel), SINGLE, f"single_run_{acf}{plabel}_nr10_s1.png")


def main(results_dir="results", single_only=False):
    study = StudyResults.load(os.path.join(HERE, results_dir))
    if not study.runs:
        print("no results yet — run the sweep first."); return
    print(f"loaded {len(study.runs)} runs")
    if not single_only:
        save_main(study)
        save_analysis(study)
    save_single_runs(study)
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="only single_runs")
    args = ap.parse_args()
    main(single_only=args.single)
