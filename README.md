cat > README.md << 'EOF'
# Atchley Unified Framework (AUF) — Task 1

[![CI](https://github.com/devinatchley6-eng/Atchley-Unified-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/devinatchley6-eng/Atchley-Unified-Framework/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A deterministic geometry routing kernel for AI safety monitoring. Part of the Atchley Unified Framework (AUF).

## Features (Task 1)

- **7D State Vector**: Tracks [κ, ε, H, Γ, Ψ, G, Θ] (curvature, energy, entropy, rigidity, sharpness, coherence, thermal)
- **Operator Sensitivity Matrix M**: 10×7 column-stochastic matrix routing state to interventions
- **Exceedance Vector**: `z(t) = sigmoid((s - θ)/τ)` soft thresholding
- **Activation Scores**: `a(t) = M @ z(t)` per-operator activation
- **Hysteresis Selection**: N_up=3, N_down=10 prevents thrashing
- **Risk Mapping**: Conservative `max(z)` placeholder → ALLOW/EDU/THROTTLE/DEEPFREEZE

## Installation

```bash
# Clone the repository
git clone https://github.com/devinatchley6-eng/Atchley-Unified-Framework.git
cd Atchley-Unified-Framework

# Install in development mode
pip install -e ".[dev]"
