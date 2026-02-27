from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np

COORDS_7 = ["kappa", "energy", "entropy", "rigidity", "sharpness", "coherence", "thermal"]
OPS_10 = ["O_clamp", "O_contract", "O_freeze", "O_diffuse", "O_predict", "O_project", "O_recal", "O_adv", "O_recover", "O_ensemble"]

class Mode(str, Enum):
    ALLOW = "ALLOW"; EDU = "EDU"; THROTTLE = "THROTTLE"; DEEPFREEZE = "DEEPFREEZE"; SCOPE_DROP = "SCOPE_DROP"

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0); return 1.0 / (1.0 + np.exp(-x))

@dataclass(frozen=True)
class Thresholds:
    theta: Union[np.ndarray, list]; tau: Union[np.ndarray, list]
    def __post_init__(self):
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=np.float64).reshape(7))
        object.__setattr__(self, "tau", np.asarray(self.tau, dtype=np.float64).reshape(7))
        if np.any(self.tau <= 0): raise ValueError("tau must be > 0")

@dataclass(frozen=True)
class InterventionThresholds:
    theta_safe: float; theta_warn: float; theta_crit: float
    def __post_init__(self):
        if not (0.0 <= self.theta_safe <= self.theta_warn <= self.theta_crit <= 1.0): raise ValueError

@dataclass
class HysteresisConfig:
    n_up: int = 3; n_down: int = 10; theta_up: float = 0.0; theta_down: float = 0.0
    def __post_init__(self):
        if self.n_up < 1 or self.n_down < 1: raise ValueError

@dataclass
class SelectionState:
    active_operator: Optional[str] = None; above_count: int = 0; below_count: int = 0

@dataclass(frozen=True)
class GeometryOutput:
    s: np.ndarray; z: np.ndarray; a: np.ndarray; selected_operator: str; selection_reason: str; risk_layer: float; mode: Mode

def default_M() -> np.ndarray:
    M = np.zeros((10, 7))
    k, e, ent, r, sh, c, t = range(7)
    M[0, e] = 1.0; M[1, t] = 1.0; M[2, k] = 0.8; M[2, sh] = 0.2; M[3, r] = 0.9; M[3, ent] = 0.1
    M[4, c] = 0.6; M[4, sh] = 0.4; M[5, ent] = 0.7; M[5, e] = 0.3; M[6, k] = 0.34; M[6, e] = 0.33; M[6, ent] = 0.33
    M[7, sh] = 0.5; M[7, c] = 0.5; M[8, k] = 0.5; M[8, c] = 0.5; M[9, c] = 1.0
    col_sums = M.sum(axis=0)
    for j in range(7):
        if col_sums[j] > 0: M[:, j] /= col_sums[j]
    return M

class GeometryEngine:
    def __init__(self, thresholds: Thresholds, intervention_thresholds: InterventionThresholds, M: Optional[np.ndarray] = None, hysteresis: Optional[HysteresisConfig] = None):
        self.thresholds = thresholds
        self.intervention_thresholds = intervention_thresholds
        self.M = default_M() if M is None else np.asarray(M)
        if self.M.shape != (10, 7): raise ValueError
        if np.any(self.M < 0): raise ValueError
        self.hysteresis = hysteresis or HysteresisConfig()
        self.state = SelectionState()

    def compute_z(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s).reshape(7)
        return sigmoid((s - self.thresholds.theta) / self.thresholds.tau)

    def compute_a(self, z: np.ndarray) -> np.ndarray:
        return self.M @ z.reshape(7)

    def _risk_from_z(self, z): return float(np.clip(np.max(z), 0.0, 1.0))
    def _mode_from_risk(self, risk):
        th = self.intervention_thresholds
        if risk < th.theta_safe: return Mode.ALLOW
        if risk < th.theta_warn: return Mode.EDU
        if risk < th.theta_crit: return Mode.THROTTLE
        return Mode.DEEPFREEZE

    def select_operator(self, a):
        a = np.asarray(a).reshape(10)
        cand_idx = int(np.argmax(a)); cand = OPS_10[cand_idx]; cand_val = float(a[cand_idx])
        st, h = self.state, self.hysteresis

        if st.active_operator:
            active_val = float(a[OPS_10.index(st.active_operator)])
            if active_val < h.theta_down: st.below_count += 1
            else: st.below_count = 0
            if st.below_count >= h.n_down:
                prev = st.active_operator
                st.active_operator = None
                st.above_count = st.below_count = 0
                return cand, f"de-escalated from {prev}"
            return st.active_operator, f"hold: {st.active_operator}"

        if cand_val > h.theta_up: st.above_count += 1
        else: st.above_count = 0
        if st.above_count >= h.n_up:
            st.active_operator = cand
            st.below_count = 0
            return cand, f"escalated: {cand}"
        return cand, f"arming: {cand} ({st.above_count}/{h.n_up})"

    def step(self, s):
        s = np.asarray(s).reshape(7)
        z = self.compute_z(s)
        a = self.compute_a(z)
        op, reason = self.select_operator(a)
        risk = self._risk_from_z(z)
        mode = self._mode_from_risk(risk)
        return GeometryOutput(s, z, a, op, reason, risk, mode)
