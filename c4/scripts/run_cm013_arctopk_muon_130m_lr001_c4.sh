#!/usr/bin/env bash
# Compatibility entry point for the revised serial Dense -> ARC experiment.
set -euo pipefail
exec "$(dirname "$0")/run_cm014_cm015_muon_130m_lr001_c4.sh" "$@"
