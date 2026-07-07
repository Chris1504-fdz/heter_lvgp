#!/bin/bash
# Periodically refresh plots + TensorBoard event logs from results/.
# Only the `tensorboard` server is long-lived; this just appends event files.
cd /data/zhq7531/IDEAL/hetero_lvgp/study
PY=/data/zhq7531/envs/ml_gp_env/bin/python
refresh() {
  $PY analyze.py        >/dev/null 2>&1
  $PY suggest_plots.py  >/dev/null 2>&1
  $PY run_sweep.py --collect-only >/dev/null 2>&1
  $PY tb_log.py 2>&1 | grep -iE "logged condition|runs,"
}
while tmux has-session -t lvgp_study 2>/dev/null; do
  refresh
  sleep 600
done
refresh           # final refresh after the backfill ends
echo "tb_loop done"
