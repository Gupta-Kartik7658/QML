#!/usr/bin/env bash
set -e; WORKERS=${WORKERS:-6}
python run_theory_check.py --workers $WORKERS
python run_shot_budget.py --workers $WORKERS
python run_hybrid.py --workers $WORKERS
