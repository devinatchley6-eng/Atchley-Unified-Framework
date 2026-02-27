import numpy as np
import pytest
from aum.safety.geometry import GeometryEngine, Thresholds, InterventionThresholds, HysteresisConfig, OPS_10, Mode, default_M

def test_thresholds_converts_lists():
    th = Thresholds(theta=[0.1]*7, tau=[1.0]*7)
    assert isinstance(th.theta, np.ndarray) and th.theta.shape == (7,)

def test_intervention_thresholds_validation():
    with pytest.raises(ValueError): InterventionThresholds(0.6, 0.5, 0.8)

def test_default_M_properties():
    M = default_M()
    assert M.shape == (10, 7) and np.all(M >= 0) and np.allclose(M.sum(axis=0), 1.0)

def test_compute_z_monotonic():
    eng = GeometryEngine(Thresholds(np.zeros(7), np.ones(7)), InterventionThresholds(0.3, 0.6, 0.9))
    assert np.all(eng.compute_z(np.full(7, -10.0)) < eng.compute_z(np.full(7, 10.0)))

def test_hysteresis_escalation():
    eng = GeometryEngine(Thresholds(np.zeros(7), np.ones(7)), InterventionThresholds(0.3, 0.6, 0.9), hysteresis=HysteresisConfig(3, 10))
    s = np.zeros(7); s[0] = 10.0
    for _ in range(2): eng.step(s); assert eng.state.active_operator is None
    out = eng.step(s); assert eng.state.active_operator and "escalated" in out.selection_reason

def test_output_fields():
    out = GeometryEngine(Thresholds(np.zeros(7), np.ones(7)), InterventionThresholds(0.3, 0.6, 0.9)).step(np.random.randn(7))
    assert out.selected_operator in OPS_10 and out.s.shape == (7,) and out.z.shape == (7,) and out.a.shape == (10,)
