from __future__ import annotations

"""Geometry kernel for AUF - 7D state routing and operator selection."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union, List

import numpy as np
from numpy import ndarray


# Canonical coordinate order (LOCKED)
COORDS_7: List[str] = ["kappa", "energy", "entropy", "rigidity", "sharpness", "coherence", "thermal"]

# Canonical operator order (LOCKED)
OPS_10: List[str] = [
    "O_clamp",
    "O_contract",
    "O_freeze",
    "O_diffuse",
    "O_predict",
    "O_project",
    "O_recal",
    "O_adv",
    "O_recover",
    "O_ensemble",
]


class Mode(str, Enum):
    """Intervention modes for AUF core service."""

    ALLOW = "ALLOW"
    EDU = "EDU"
    THROTTLE = "THROTTLE"
    DEEPFREEZE = "DEEPFREEZE"
    SCOPE_DROP = "SCOPE_DROP"


def sigmoid(x: ndarray) -> ndarray:
    """Stable sigmoid function."""
    x_clipped: ndarray = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


@dataclass(frozen=True)
class Thresholds:
    """Per-coordinate thresholds theta and softness tau."""

    theta: Union[ndarray, List[float]]
    tau: Union[ndarray, List[float]]

    def __post_init__(self) -> None:
        theta_arr: ndarray = np.asarray(self.theta, dtype=np.float64).reshape(7)
        tau_arr: ndarray = np.asarray(self.tau, dtype=np.float64).reshape(7)

        object.__setattr__(self, "theta", theta_arr)
        object.__setattr__(self, "tau", tau_arr)

        if np.any(tau_arr <= 0):
            raise ValueError("tau must be > 0 for all coordinates")


@dataclass(frozen=True)
class InterventionThresholds:
    """Scalar thresholds for risk -> mode mapping."""

    theta_safe: float
    theta_warn: float
    theta_crit: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.theta_safe <= self.theta_warn <= self.theta_crit <= 1.0):
            raise ValueError("Require 0 <= safe <= warn <= crit <= 1")


@dataclass
class HysteresisConfig:
    """Configuration for operator selection hysteresis."""

    n_up: int = 3
    n_down: int = 10
    theta_up: float = 0.0
    theta_down: float = 0.0

    def __post_init__(self) -> None:
        if self.n_up < 1 or self.n_down < 1:
            raise ValueError("n_up and n_down must be >= 1")


@dataclass
class SelectionState:
    """Maintains hysteresis state across steps."""

    active_operator: Optional[str] = None
    above_count: int = 0
    below_count: int = 0


@dataclass(frozen=True)
class GeometryOutput:
    """Complete output from geometry engine."""

    s: ndarray
    z: ndarray
    a: ndarray
    selected_operator: str
    selection_reason: str
    risk_layer: float
    mode: Mode


def default_M() -> ndarray:
    """Default operator sensitivity matrix M (10×7)."""
    M: ndarray = np.zeros((10, 7), dtype=np.float64)

    # Indices for readability
    kappa, energy, entropy, rigidity, sharpness, coherence, thermal = range(7)

    # O_clamp: energy excess
    M[0, energy] = 1.0

    # O_contract: thermal excess
    M[1, thermal] = 1.0

    # O_freeze: curvature spike + sharpness
    M[2, kappa] = 0.8
    M[2, sharpness] = 0.2

    # O_diffuse: high rigidity -> explore; low entropy
    M[3, rigidity] = 0.9
    M[3, entropy] = 0.1

    # O_predict: coherence + sharpness
    M[4, coherence] = 0.6
    M[4, sharpness] = 0.4

    # O_project: entropy + energy
    M[5, entropy] = 0.7
    M[5, energy] = 0.3

    # O_recal: drift across kappa, energy, entropy
    M[6, kappa] = 0.34
    M[6, energy] = 0.33
    M[6, entropy] = 0.33

    # O_adv: escalation - sharpness + coherence
    M[7, sharpness] = 0.5
    M[7, coherence] = 0.5

    # O_recover: sustained violation - curvature + coherence
    M[8, kappa] = 0.5
    M[8, coherence] = 0.5

    # O_ensemble: stabilize coherence
    M[9, coherence] = 1.0

    # Column-stochastic normalization
    col_sums: ndarray = M.sum(axis=0)
    for j in range(M.shape[1]):
        if col_sums[j] > 0:
            M[:, j] /= col_sums[j]

    return M


class GeometryEngine:
    """Core geometry engine for state routing and operator selection."""

    def __init__(
        self,
        thresholds: Thresholds,
        intervention_thresholds: InterventionThresholds,
        M: Optional[ndarray] = None,
        hysteresis: Optional[HysteresisConfig] = None,
    ) -> None:
        self.thresholds = thresholds
        self.intervention_thresholds = intervention_thresholds
        self.M: ndarray = default_M() if M is None else np.asarray(M, dtype=np.float64)

        if self.M.shape != (10, 7):
            raise ValueError("M must be shape (10,7)")
        if np.any(self.M < 0):
            raise ValueError("M must be non-negative")

        self.hysteresis = hysteresis or HysteresisConfig()
        self.state = SelectionState()

    def compute_z(self, s: ndarray) -> ndarray:
        """Compute exceedance vector z(t) = sigmoid((s - theta)/tau)."""
        s_reshaped: ndarray = np.asarray(s, dtype=np.float64).reshape(7)
        return sigmoid((s_reshaped - self.thresholds.theta) / self.thresholds.tau)

    def compute_a(self, z: ndarray) -> ndarray:
        """Compute activation scores a(t) = M @ z(t)."""
        z_reshaped: ndarray = np.asarray(z, dtype=np.float64).reshape(7)
        return self.M @ z_reshaped

    def _risk_from_z(self, z: ndarray) -> float:
        """Placeholder for full layer risk calculation."""
        return float(np.clip(np.max(z), 0.0, 1.0))

    def _mode_from_risk(self, risk: float) -> Mode:
        """Map risk score to intervention mode."""
        if risk < self.intervention_thresholds.theta_safe:
            return Mode.ALLOW
        if risk < self.intervention_thresholds.theta_warn:
            return Mode.EDU
        if risk < self.intervention_thresholds.theta_crit:
            return Mode.THROTTLE
        return Mode.DEEPFREEZE

    def select_operator(self, a: ndarray) -> Tuple[str, str]:
        """Select operator with hysteresis guard."""
        a_reshaped: ndarray = np.asarray(a, dtype=np.float64).reshape(10)
        cand_idx: int = int(np.argmax(a_reshaped))
        cand: str = OPS_10[cand_idx]
        cand_val: float = float(a_reshaped[cand_idx])

        st: SelectionState = self.state
        h: HysteresisConfig = self.hysteresis

        # De-escalation check for current active operator
        if st.active_operator is not None:
            active_idx: int = OPS_10.index(st.active_operator)
            active_val: float = float(a_reshaped[active_idx])

            if active_val < h.theta_down:
                st.below_count += 1
            else:
                st.below_count = 0

            if st.below_count >= h.n_down:
                prev: str = st.active_operator
                st.active_operator = None
                st.above_count = 0
                st.below_count = 0
                return cand, f"de-escalated from {prev}"

            return st.active_operator, f"hold: {st.active_operator}"

        # No active operator - arm candidate
        if cand_val > h.theta_up:
            st.above_count += 1
        else:
            st.above_count = 0

        if st.above_count >= h.n_up:
            st.active_operator = cand
            st.below_count = 0
            return cand, f"escalated: {cand}"

        return cand, f"arming: {cand} ({st.above_count}/{h.n_up})"

    def step(self, s: ndarray) -> GeometryOutput:
        """Process one timestep: compute all quantities."""
        s_reshaped: ndarray = np.asarray(s, dtype=np.float64).reshape(7)

        z: ndarray = self.compute_z(s_reshaped)
        a: ndarray = self.compute_a(z)
        op: str
        reason: str
        op, reason = self.select_operator(a)

        risk: float = self._risk_from_z(z)
        mode: Mode = self._mode_from_risk(risk)

        return GeometryOutput(
            s=s_reshaped,
            z=z,
            a=a,
            selected_operator=op,
            selection_reason=reason,
            risk_layer=risk,
            mode=mode,
        )
