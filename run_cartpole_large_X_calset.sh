#!/usr/bin/env bash
set -euo pipefail

# Run CartPole training using the configargparse config file.
# This avoids bash line-continuation issues when copy/pasting long commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python run_experiment.py -c configs/cartpole_large_X_calset.cfg --mode train

