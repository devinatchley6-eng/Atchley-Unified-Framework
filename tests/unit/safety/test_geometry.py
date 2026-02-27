from __future__ import annotations

import numpy as np
import pytest
from aum.safety.geometry import (
    GeometryEngine,
    Thresholds,
    InterventionThresholds,
    HysteresisConfig,
    OPS_10,
    Mode,
    default_M,
)


def test_thresholds_converts_lists() -> None:
    th = Thresholds(theta=[0.1] * 7, tau=[1.0] * 7)
    assert isinstance(th.theta, np.ndarray)
    assert th.theta.shape == (7,)


def test_intervention_thresholds_validation() -> None:
    with pytest.raises(ValueError):
        InterventionThresholds(0.6, 0.5, 0.8)


def test_default_M_properties() -> None:
    M = default_M()
    assert M.shape == (10, 7)
    assert np.all(M >= 0)
    assert np.allclose(M.sum(axis=0), 1.0)


def test_compute_z_monotonic() -> None:
    eng = GeometryEngine(
        Thresholds(np.zeros(7), np.ones(7)),
        InterventionThresholds(0.3, 0.6, 0.9),
    )
    z_low = eng.compute_z(np.full(7, -10.0))
    z_high = eng.compute_z(np.full(7, 10.0))
    assert np.all(z_low < z_high)
    assert np.all((0.0 <= z_low) & (z_low <= 1.0))


def test_hysteresis_escalation() -> None:
    eng = GeometryEngine(
        Thresholds(np.zeros(7), np.ones(7)),
        InterventionThresholds(0.3, 0.6, 0.9),
        hysteresis=HysteresisConfig(3, 10),
    )
    s = np.zeros(7)
    s[0] = 10.0

    for _ in range(2):
        eng.step(s)
        assert eng.state.active_operator is None

    out = eng.step(s)
    assert eng.state.active_operator is not None
    assert "escalated" in out.selection_reason


def test_output_fields() -> None:
    out = GeometryEngine(
        Thresholds(np.zeros(7), np.ones(7)),
        InterventionThresholds(0.3, 0.6, 0.9),
    ).step(np.random.randn(7))

    assert out.selected_operator in OPS_10
    assert out.s.shape == (7,)
    assert out.z.shape == (7,)
    assert out.a.shape == (10,)
    assert 0.0 <= out.risk_layer <= 1.0
