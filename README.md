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
#Clone the repository
git clone https://github.com/devinatchley6-eng/Atchley-Unified-Framework.git
cd Atchley-Unified-Framework

# Install in development mode
pip install -e ".[dev]"
import numpy as np
from aum.safety.geometry import GeometryEngine, Thresholds, InterventionThresholds

# Initialize with default thresholds
engine = GeometryEngine(
    thresholds=Thresholds(theta=np.zeros(7), tau=np.ones(7)),
    intervention_thresholds=InterventionThresholds(0.3, 0.6, 0.9)
)

# Process a state vector
state = np.array([0.5, 0.2, 0.8, 0.1, 0.3, 0.6, 0.4])
output = engine.step(state)

print(f"Selected operator: {output.selected_operator}")
print(f"Risk level: {output.risk_layer}")
print(f"Intervention mode: {output.mode}")
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=aum tests/
src/
└── aum/
    └── safety/
        ├── __init__.py
        └── geometry.py      # Core routing kernel
tests/
└── unit/
    └── safety/
        └── test_geometry.py # Comprehensive tests

## 2. Create a Release

### Option A: Create release via GitHub web UI (easiest)

1. Go to: https://github.com/devinatchley6-eng/Atchley-Unified-Framework/releases
2. Click **"Create a new release"**
3. **Tag version:** `v0.1.0`
4. **Release title:** `Task 1: Geometry Routing Kernel`
5. **Description:**
   ```markdown
   ## 🎉 First official release of AUF Task 1
   
   ### Features
   - ✅ 7D state vector [κ, ε, H, Γ, Ψ, G, Θ]
   - ✅ Operator sensitivity matrix M (10×7)
   - ✅ Exceedance vector z(t) with soft thresholding
   - ✅ Activation scores a(t) = M @ z(t)
   - ✅ Hysteresis operator selection (n_up=3, n_down=10)
   - ✅ Risk-based mode mapping (ALLOW/EDU/THROTTLE/DEEPFREEZE)
   
   ### Testing
   - ✅ 6/6 unit tests passing
   - ✅ Full CI pipeline with ruff, mypy, pytest
   
   ### Installation
   ```bash
   pip install git+https://github.com/devinatchley6-eng/Atchley-Unified-Framework.git@v0.1.0
