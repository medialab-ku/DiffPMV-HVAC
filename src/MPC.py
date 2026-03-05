# src/mpc.py
# -----------------------------------------------------------------------------
# Explicit (linearized) PMV-MPC controller for src.
#
# - Uses a low-order discrete-time thermal model of the ROI temperature
# - Linearizes the Fanger PMV index around a comfort point (T0, v0)
# - Solves a receding-horizon QP at each time-step using cvxpy
# - Outputs a control sequence compatible with src: (u, ut, phi)
#
# This code is "explicit model-based" MPC (uses an explicit model of the
# dynamics and comfort index), but the QP itself is still solved online.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math as _py_math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import cvxpy as cp
from tqdm import tqdm

# Pull in user's codebase (phi.torch, torch, CenteredGrid, etc.)
from src import torch, field, math, CenteredGrid
from src.env import Env
from src.forward import Forward


# ==================== Data classes / configs ====================

@dataclass
class DynParams:
    """
    Simplified RC thermal network model for ROI-averaged air temperature.

    Discrete-time state-space model:
        T_{k+1} = a_T * T_k + b_u * u_k + b_ut * ut_k + b_uT * u_k * (ut_k - T_k)
                  + b_occ * Q_occ,k + b_ext * T_ext,k + c_0

    Physical interpretation:
        - a_T:   thermal retention coefficient (≈ exp(-Δt/τ), τ = thermal time constant)
        - b_u:   direct impact of supply air velocity on mixing/convection
        - b_ut:  direct impact of supply air temperature
        - b_uT:  bilinear coupling term (heat transfer effectiveness)
        - b_occ: internal heat gains from occupants [K per occupant]
        - b_ext: heat exchange with external/surrounding zones
        - c_0:   bias term (accounts for baseline heat sources)

    Reference: Ferkl & Široký (2010) "Ceiling radiant cooling: Comparison of ARMAX
               and subspace identification modelling methods"

    NOTE: These parameters must be identified from CFD simulation data or
          real measurements via system identification (e.g., least squares regression).
    """
    # Thermal dynamics
    a_T: float = 0.92      # thermal retention (higher = slower temperature change)
    b_u: float = 0.01      # velocity mixing effect
    b_ut: float = 0.05     # supply temperature direct effect
    b_uT: float = 0.02     # bilinear heat transfer coupling

    # Disturbances
    b_occ: float = 0.01    # internal gains per occupant [K/person]
    b_ext: float = 0.02    # external temperature influence
    c_0: float = 0.0       # constant offset


@dataclass
class CostParams:
    """
    Multi-objective cost function weights for MPC optimization.

    Objective function:
        J = Σ_k [ w_pmv * Σ_i w_i * PMV_i,k²
                + w_u * u_k²
                + w_ut * (ut_k - ut_ref)²
                + w_du * |u_k - u_{k-1}|
                + w_dut * |ut_k - ut_{k-1}| ]

    where:
        - PMV_i,k: linearized PMV for occupant i at time k
        - w_i: individual occupant weight (priority)
        - u_k: supply air velocity [m/s]
        - ut_k: supply air temperature [°C]

    Reference: Široký et al. (2011) "Experimental analysis of model predictive
               control for an energy efficient building heating system"
    """
    # Thermal comfort (primary objective)
    w_pmv: float = 10.0        # PMV deviation penalty (higher = prioritize comfort)

    # Energy consumption
    w_u: float = 1.0           # fan power ∝ u³, quadratic approximation
    w_ut: float = 1.0          # heating/cooling energy ∝ |ut - T_ext|
    ut_ref: float = 14.4       # reference supply temperature [°C] (energy-optimal)

    # Control smoothness (actuator wear, occupant disturbance)
    w_du: float = 0.1          # velocity change penalty
    w_dut: float = 0.1         # temperature change penalty


@dataclass
class Bounds:
    # control bounds
    u_min: float = 0.1
    u_max: float = 2.00
    ut_min: float = 10.0
    ut_max: float = 22.0

    # thermal comfort / safety bounds
    T_min: float = 16.0
    T_max: float = 30.0

    # diffuser angle (what Forward.step() expects as phi)
    phi_min: float = 0.0
    phi_max: float = _py_math.pi / 2.0   # up to 90 deg tilt


@dataclass
class PMVParams:
    """
    Fanger PMV model parameters and comfort constraints.

    PMV (Predicted Mean Vote) scale:
        -3: cold, -2: cool, -1: slightly cool,
         0: neutral (optimal comfort),
        +1: slightly warm, +2: warm, +3: hot

    Linearization:
        PMV(T, v) ≈ a_T * T + a_v * v + a_0
    where coefficients (a_T, a_v, a_0) are computed via finite differences
    around the operating point (T0, v0) and updated at each MPC step.

    Reference: ISO 7730:2005 "Ergonomics of the thermal environment"
    """
    rh: float = 50.0           # relative humidity [%]
    met: float = 1.0           # metabolic rate [met] (1.0 = sedentary, 1.2 = light work)
    clo: float = 0.74          # clothing insulation [clo] (0.5 = summer, 1.0 = winter)
    pmv_band: float = 0.5      # acceptable PMV range: [-pmv_band, +pmv_band]
                               # ASHRAE 55: ±0.5 (80% satisfaction), ±0.7 (90%)


@dataclass
class MPCConfig:
    """
    MPC algorithm configuration and implementation settings.

    Key algorithmic choices:
        1. Model reduction: 3D CFD → scalar temperature via ROI averaging
        2. PMV linearization: Updated each step around current state
        3. Convexification: Fixed diffuser angle φ (non-convex if optimized)
        4. Constraint handling: Soft (penalty) or hard (feasibility) PMV bounds

    Reference: Prívara et al. (2011) "Model predictive control of a building
               heating system: The first experience"
    """
    # State aggregation from CFD field
    use_roi_avg: bool = True               # True: weighted average over occupant ROIs
                                           # False: spatial mean over entire domain

    # Multi-occupant handling
    enforce_occ_indices: Optional[List[int]] = None  # None = all occupants
                                                      # List = specific subset

    # PMV linearization operating point (updated online during simulation)
    pmv_linearize_T0: float = 23.0         # nominal temperature [°C]
    pmv_linearize_v0: float = 0.15         # nominal air velocity [m/s]
    update_linearization: bool = True      # True: recompute at each step (tracking)
                                           # False: fixed coefficients (faster but less accurate)

    # Constraint enforcement strategy
    use_pmv_band_constraint: bool = False  # False: soft penalty in objective
                                           # True: hard constraint (may cause infeasibility)

    # Diffuser control (non-convex if decision variable)
    phi_nominal: float = 0.0               # fixed angle [rad] (0 = horizontal)
                                           # Future work: adaptive φ via MINLP

    # QP solver selection
    cvxpy_solver: str = "OSQP"             # Options: OSQP (robust), ECOS (fast), GUROBI (commercial)


# ==================== PMV utilities ====================

def _compute_pmv_scalar(T: float,
                        v: float,
                        rh: float,
                        met: float,
                        clo: float) -> float:
    """
    Scalar Fanger PMV (same equations as Forward.tensor_pmv, but working on
    Python floats / scalars).

    T  : air temperature [°C]
    v  : air velocity [m/s]
    rh : relative humidity [%]
    met: metabolic rate [met units]
    clo: clothing insulation [clo units]
    tr : mean radiant temperature = T + 0.4 [°C]
    """
    import math

    # water vapor partial pressure [Pa]
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (T + 235.0))

    # constants
    M   = met * 58.15          # metabolic rate [W/m^2]
    W   = 0.0                  # external work [W/m^2]
    icl = clo * 0.155          # clothing insulation [m^2 K/W]

    # clothing area factor (forward.py와 동일)
    if icl < 0.078:
        f_cl = 1.00 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    # convective heat transfer coefficient
    hcf = 12.1 * math.sqrt(max(v, 1e-8))  # avoid zero
    hc = hcf  # initialize

    # mean radiant temperature (forward.py와 동일: tr = T + 0.4)
    tr = T + 0.4
    tra = tr + 273.0
    taa = T + 273.0

    # initial clothing surface temp guess
    t_cla = taa + (35.5 - T) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * (M - W)) + p2 * (tra / 100.0) ** 4

    xn  = t_cla / 100.0
    xf  = t_cla / 50.0
    tol = 0.00015  # forward.py와 동일

    # Iterative solution for clothing surface temperature (forward.py와 동일)
    for _ in range(150):
        xf_prev = xf
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)

        if abs(xn - xf) < tol:
            break
        xf = 0.5 * (xf_prev + xn)

    tcl = 100.0 * xn - 273.0

    # heat loss terms (forward.py와 동일)
    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * (M - W) - pa)
    hl2 = max(0.42 * ((M - W) - 58.15), 0.0)
    hl3 = 1.7 * 0.00001 * M * (5867.0 - pa)
    hl4 = 0.0014 * M * (34.0 - T)
    hl5 = 3.96 * f_cl * (xn ** 4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - T)

    ts = 0.303 * math.exp(-0.036 * M) + 0.028
    pmv = ts * ((M - W) - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return float(pmv)


def _finite_diff_pmv_linearization(T0: float,
                                   v0: float,
                                   rh: float,
                                   met: float,
                                   clo: float,
                                   dT: float = 0.1,
                                   dv: float = 0.05):
    """
    Finite-difference linearization of PMV around (T0, v0).

    Returns coefficients a_T, a_v, a0 s.t.

        PMV(T, v) ≈ a_T * T + a_v * v + a0
    """
    pmv0   = _compute_pmv_scalar(T0, v0, rh, met, clo)
    pmv_Tp = _compute_pmv_scalar(T0 + dT, v0, rh, met, clo)
    pmv_Tm = _compute_pmv_scalar(T0 - dT, v0, rh, met, clo)
    a_T    = (pmv_Tp - pmv_Tm) / (2.0 * dT)

    pmv_vp = _compute_pmv_scalar(T0, v0 + dv, rh, met, clo)
    pmv_vm = _compute_pmv_scalar(T0, v0 - dv, rh, met, clo)
    a_v    = (pmv_vp - pmv_vm) / (2.0 * dv)

    a0     = pmv0 - a_T * T0 - a_v * v0
    return a_T, a_v, a0, pmv0


# ==================== MPC controller ====================

class PMVExplicitMPC:
    """
    Model Predictive Control (MPC) for PMV-based thermal comfort optimization.

    ============================================================================
    ALGORITHM OVERVIEW (Receding Horizon Control)
    ============================================================================

    At each discrete time step t = 0, 1, ..., Nt-1:

    1. STATE MEASUREMENT
       - Extract scalar ROI temperature T_t from 3D CFD temperature field
       - Method: Weighted average over occupant regions (Gaussian kernels)

    2. PREDICTION MODEL
       - Simplified RC thermal dynamics (DynParams):
         T_{k+1} = f(T_k, u_k, ut_k, disturbances)
       - Linearized bilinear terms for QP convexity

    3. PMV LINEARIZATION
       - Update coefficients around current state (T_t, v_t):
         PMV(T, v) ≈ a_T * T + a_v * v + a_0
       - Finite-difference approximation of ∂PMV/∂T, ∂PMV/∂v

    4. OPTIMIZATION (Quadratic Program)
       - Decision variables: u[0:Np], ut[0:Np], T[0:Np+1]
       - Objective: min J_comfort + J_energy + J_smoothness
       - Constraints: dynamics, bounds, optional PMV band
       - Solver: OSQP (Interior Point Method)

    5. CONTROL APPLICATION
       - Apply ONLY first control: (u_0, ut_0, φ_nominal)
       - Advance CFD simulation: (v, T) ← Forward.step(v, T, controls)
       - Discard remaining horizon predictions

    6. REPEAT
       - Shift time window and resolve at t+1

    ============================================================================
    THEORETICAL BACKGROUND
    ============================================================================

    This implementation follows the methodology of:

    [1] Široký, J., et al. (2011). "Experimental analysis of model predictive
        control for an energy efficient building heating system."
        Applied Energy, 88(9), 3079-3087.

    [2] Prívara, S., et al. (2013). "Building modeling as a crucial part for
        building predictive control." Energy and Buildings, 56, 8-22.

    [3] Oldewurtel, F., et al. (2012). "Use of model predictive control and
        weather forecasts for energy efficient building climate control."
        Energy and Buildings, 45, 15-27.

    Key algorithmic features:
    - Linearized PMV for QP formulation (vs. nonlinear PMV ⇒ NLP)
    - Online coefficient update (vs. fixed linearization ⇒ poor tracking)
    - Soft comfort constraints (vs. hard constraints ⇒ infeasibility risk)
    - Bilinear dynamics approximation (vs. full CFD ⇒ computational cost)

    ============================================================================
    OUTPUT
    ============================================================================

    control_vars : torch.Tensor, shape (Nt, 3)
        Time series of optimal controls on env.device:
        control_vars[t, 0] = u_t    [m/s]  supply air velocity
        control_vars[t, 1] = ut_t   [°C]   supply air temperature
        control_vars[t, 2] = phi_t  [rad]  diffuser angle (fixed)

    ============================================================================
    """

    def __init__(self,
                 env: Env,
                 horizon: int,
                 dyn: DynParams,
                 cost: CostParams,
                 bounds: Bounds,
                 pmv: PMVParams,
                 cfg: MPCConfig):

        self.env = env
        self.fwd = Forward(env)
        self.Np  = int(horizon)

        self.dyn = dyn
        self.cost = cost
        self.bnd = bounds
        self.pmv = pmv
        self.cfg = cfg

        # Which occupants we care about explicitly in the MPC cost
        if cfg.enforce_occ_indices:
            indices = [i for i in cfg.enforce_occ_indices if 0 <= i < len(env.occs)]
        else:
            indices = list(range(len(env.occs)))

        self._occ_inds: List[int] = indices

        # Pre-compute PMV linearization coefficients for each enforced occupant
        self._pmv_coefs: List[Tuple[float, float, float]] = []
        self._occ_weights: np.ndarray = np.zeros(len(indices), dtype=float)

        for local_idx, occ_idx in enumerate(indices):
            occ = env.occs[occ_idx]
            met = getattr(occ, "met", pmv.met)
            clo = getattr(occ, "clo", pmv.clo)
            aT, aV, a0, _ = _finite_diff_pmv_linearization(
                cfg.pmv_linearize_T0,
                cfg.pmv_linearize_v0,
                pmv.rh,
                met,
                clo,
            )
            self._pmv_coefs.append((aT, aV, a0))
            self._occ_weights[local_idx] = getattr(occ, "weight", 1.0)

        self._n_occ = len(self._pmv_coefs)

    # ---------- helpers ----------

    def _scalar_T(self, temperature: CenteredGrid, t_idx: int) -> float:
        """
        Compute a scalar 'room temperature' from the full CFD temperature field.

        If use_roi_avg is True and occupant kernels are available, we compute
        a weighted average of ROI temperatures (using occ.weight as weights).
        Otherwise we fall back to the global mean.
        """
        if self.cfg.use_roi_avg and len(self.env.occs) > 0:
            num = 0.0
            den = 0.0
            for occ in self.env.occs:
                # occupancies were pre-built in Env._setROIKernel()
                if not hasattr(occ, "kernels"):
                    continue
                ker = occ.kernels.t[t_idx]
                num_i = field.l1_loss(ker * temperature).native().item()
                den_i = field.l1_loss(ker).native().item() + 1e-8
                w_i = getattr(occ, "weight", 1.0)
                num += w_i * (num_i / den_i)
                den += w_i
            if den > 0.0:
                return float(num / den)

        # Fallback: global mean temperature in the domain
        return float(field.mean(temperature).native().item())

    def _occ_count(self, t_idx: int) -> float:
        """
        Number of active occupants at time index t_idx.
        """
        return float(sum(1 for occ in self.env.occs if occ.is_active(t_idx)))

    def _occ_active_matrix(self, t0: int) -> np.ndarray:
        """
        Returns a matrix active[local_idx, k] in {0,1} indicating whether
        enforced occupant 'local_idx' is active at global time t = t0 + k.
        """
        active = np.zeros((self._n_occ, self.Np), dtype=float)
        for local_idx, occ_idx in enumerate(self._occ_inds):
            occ = self.env.occs[occ_idx]
            for k in range(self.Np):
                t = t0 + k
                active[local_idx, k] = 1.0 if occ.is_active(t) else 0.0
        return active

    # ---------- QP builder ----------

    def _solve_qp(self,
                  T0: float,
                  Tout_seq: np.ndarray,
                  occ_count_seq: np.ndarray,
                  occ_active_seq: np.ndarray,
                  u_prev: Optional[float],
                  ut_prev: Optional[float],
                  u_bar: float,
                  ut_bar: float,
                  T_bar: float) -> Tuple[float, float]:
        """
        Build and solve the horizon QP for u[0:Np], ut[0:Np], T[0:Np+1] and
        return only the first control (u0, ut0).

        PMV is treated as a soft objective:

          sum_k sum_occ  w_occ * w_pmv * | PMV_lin(T_k, v_k) |^2

        with v_k ≈ ku * u_k. Optionally, PMV band constraints can also be
        enforced (cfg.use_pmv_band_constraint).
        """
        Np = self.Np
        n_occ = self._n_occ

        # decision variables
        u  = cp.Variable(Np)
        ut = cp.Variable(Np)
        T  = cp.Variable(Np + 1)
        du = cp.Variable(Np)

        cons = [T[0] == T0]

        pmv_cost = 0.0

        for k in range(Np):
            # ========== Thermal Dynamics Constraint ==========
            # Simplified RC model with bilinear heat transfer term.
            # Reference: Eq. (3) in Ferkl & Široký (2010)
            #
            # Exact dynamics (nonlinear):
            #   T_{k+1} = a_T*T_k + b_ut*ut_k + b_uT*u_k*(ut_k - T_k) + disturbances
            #
            # Linearization around (u_bar, ut_bar, T_bar):
            #   u_k*(ut_k - T_k) ≈ (ut_bar - T_bar)*u_k + u_bar*ut_k - u_bar*T_k
            #                      - u_bar*(ut_bar - T_bar)
            #
            # This first-order Taylor expansion preserves convexity of the QP.

            bilinear_approx = (
                (ut_bar - T_bar) * u[k] +      # ∂(u·(ut-T))/∂u term
                u_bar * ut[k] -                # ∂(u·(ut-T))/∂ut term
                u_bar * T[k] -                 # ∂(u·(ut-T))/∂T term
                u_bar * (ut_bar - T_bar)       # remove double-counted baseline
            )

            cons.append(
                T[k + 1] ==
                self.dyn.a_T * T[k] +                          # thermal inertia
                self.dyn.b_u * u[k] +                          # mixing effect
                self.dyn.b_ut * ut[k] +                        # supply temp
                self.dyn.b_uT * bilinear_approx +              # heat transfer
                self.dyn.b_occ * occ_count_seq[k] +            # occupant gains
                self.dyn.b_ext * (Tout_seq[k] - T[k]) +        # external exchange
                self.dyn.c_0                                   # constant offset
            )

            # temperature bounds at step k
            cons += [self.bnd.T_min <= T[k], T[k] <= self.bnd.T_max]

            # control bounds
            cons += [self.bnd.u_min <= u[k], u[k] <= self.bnd.u_max]
            cons += [self.bnd.ut_min <= ut[k], ut[k] <= self.bnd.ut_max]

            # control rate (Δu) as L1 epigraph
            if u_prev is None and k == 0:
                cons += [du[k] >= u[k], du[k] >= -u[k]]
            elif k == 0:
                cons += [du[k] >= u[k] - u_prev, du[k] >= -(u[k] - u_prev)]
            else:
                cons += [du[k] >= u[k] - u[k - 1], du[k] >= -(u[k] - u[k - 1])]

            # ========== PMV Thermal Comfort Cost ==========
            # Linearized PMV around current operating point (T_bar, v_bar).
            # Reference: Section III.B in Široký et al. (2011)
            #
            # Full Fanger PMV is nonlinear: PMV = f(T, v, rh, met, clo, tr)
            # Linearization: PMV(T,v) ≈ a_T*T + a_v*v + a_0
            #
            # Coefficients (a_T, a_v, a_0) computed via finite differences and
            # updated at each MPC iteration to track the current operating regime.
            #
            # Assumption: v ≈ u (supply velocity equals ROI velocity)
            # This is an approximation; actual v depends on distance from diffuser
            # and room geometry. Could be refined with v = κ*u, κ identified from CFD.

            for local_idx in range(n_occ):
                active = occ_active_seq[local_idx, k]
                if active <= 0.0:
                    continue  # occupant not present at this time step

                aT_i, aV_i, a0_i = self._pmv_coefs[local_idx]
                w_occ = self._occ_weights[local_idx]

                # Linearized PMV: assumes v_ROI ≈ u_supply
                pmv_lin = aT_i * T[k] + aV_i * u[k] + a0_i

                # Soft penalty on squared PMV deviation from neutral (PMV=0)
                # Reference: Eq. (5) in Goyal et al. (2013) "Occupancy-based HVAC control"
                pmv_cost += (
                    self.cost.w_pmv *
                    float(active) *
                    float(w_occ) *
                    cp.square(pmv_lin)
                )

                # Optional hard comfort constraint: |PMV| ≤ pmv_band
                # Pro: guarantees comfort. Con: may cause infeasibility if disturbances large
                if self.cfg.use_pmv_band_constraint:
                    cons += [pmv_lin <= self.pmv.pmv_band]
                    cons += [pmv_lin >= -self.pmv.pmv_band]

        # ========== Terminal Constraint ==========
        # Ensures temperature remains safe at end of prediction horizon
        cons += [self.bnd.T_min <= T[Np], T[Np] <= self.bnd.T_max]

        # ========== Objective Function ==========
        # Multi-objective cost balancing comfort, energy, and smoothness.
        # Reference: Eq. (4) in Oldewurtel et al. (2012) "Use of model predictive
        #            control and weather forecasts for energy efficient building climate control"
        #
        # J = J_comfort + J_energy + J_smoothness
        #
        # where:
        #   J_comfort    = Σ w_pmv * PMV²  (already accumulated in pmv_cost)
        #   J_energy     = Σ [w_u*u² + w_ut*(ut - ut_ref)²]
        #   J_smoothness = Σ [w_du*|Δu| + w_dut*|Δut|]
        #
        # Note: Fan power ∝ u³ (affinity laws), approximated as quadratic for QP convexity.
        #       Heating/cooling power ∝ mass_flow * ΔT ≈ u * |ut - T_ext|.

        dut = cp.Variable(Np)  # auxiliary variable for |ut[k] - ut[k-1]|

        # Temperature change smoothness constraints
        ut_prev_val = ut_prev if ut_prev is not None else self.cost.ut_ref
        for k in range(Np):
            if k == 0:
                cons += [dut[k] >= ut[k] - ut_prev_val, dut[k] >= -(ut[k] - ut_prev_val)]
            else:
                cons += [dut[k] >= ut[k] - ut[k-1], dut[k] >= -(ut[k] - ut[k-1])]

        obj = (
            pmv_cost                                              # thermal comfort
            + self.cost.w_u * cp.sum_squares(u)                   # fan energy (∝ u³ ≈ u²)
            + self.cost.w_ut * cp.sum_squares(ut - self.cost.ut_ref)  # heating/cooling energy
            + self.cost.w_du * cp.sum(du)                         # velocity smoothness
            + self.cost.w_dut * cp.sum(dut)                       # temperature smoothness
        )

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=self.cfg.cvxpy_solver, warm_start=True, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC QP infeasible: {prob.status}")

        return float(u.value[0]), float(ut.value[0])

    # ---------- Main loop over env.Nt ----------

    def run(self) -> "torch.Tensor":
        """
        Execute receding-horizon MPC over env.Nt steps on the user's PDE
        simulator.

        Returns
        -------
        control_vars : torch.Tensor, shape (Nt, 3)
            control_vars[t] = (u_t, ut_t, phi_t) on env.device.
        """
        Nt = self.env.Nt
        device = self.env.device

        control_vars = torch.zeros((Nt, 3), device=device, dtype=torch.float32)

        # PDE state
        vel = self.env.init_vf
        tem = self.env.init_tf

        # previous controls for smoothing
        u_prev: Optional[float] = None
        ut_prev: Optional[float] = None

        for t in tqdm(range(Nt), desc="MPC Steps", unit="step"):
            # ========== Step 1: State Measurement ==========
            # Extract scalar temperature from 3D CFD field
            T_now = self._scalar_T(tem, t_idx=t)

            # ========== Step 2: Update PMV Linearization (Optional) ==========
            # Recompute linearization coefficients around current state for better tracking
            if self.cfg.update_linearization and t > 0:
                # Current velocity estimate (from previous control)
                v_now = float(u_prev) if u_prev is not None else self.cfg.pmv_linearize_v0

                # Update linearization for each occupant
                for local_idx, occ_idx in enumerate(self._occ_inds):
                    occ = self.env.occs[occ_idx]
                    met = getattr(occ, "met", self.pmv.met)
                    clo = getattr(occ, "clo", self.pmv.clo)
                    aT, aV, a0, _ = _finite_diff_pmv_linearization(
                        T_now,      # Current temperature (updated!)
                        v_now,      # Current velocity (updated!)
                        self.pmv.rh,
                        met,
                        clo,
                    )
                    self._pmv_coefs[local_idx] = (aT, aV, a0)

            # ========== Step 3: Build Disturbance Predictions ==========
            # (here we use "constant hold" for Tout, and discrete occupancy count)
            if hasattr(self.env.scene, "init_room_temp"):
                Tout_now = float(self.env.scene.init_room_temp)
            else:
                Tout_now = T_now

            Tout_seq = np.full(self.Np, Tout_now, dtype=float)
            occ_count_seq = np.array(
                [self._occ_count(t + k) for k in range(self.Np)],
                dtype=float,
            )
            occ_active_seq = self._occ_active_matrix(t0=t)

            # ========== Step 4: Choose Linearization Point for Bilinear Dynamics ==========
            if u_prev is None:
                u_bar = float((self.bnd.u_min + self.bnd.u_max) / 2.0)
            else:
                u_bar = float(u_prev)

            if ut_prev is None:
                ut_bar = float((self.bnd.ut_min + self.bnd.ut_max) / 2.0)
            else:
                ut_bar = float(ut_prev)

            # ========== Step 5: Solve MPC Optimization (QP) ==========
            try:
                u0, ut0 = self._solve_qp(
                    T0=T_now,
                    Tout_seq=Tout_seq,
                    occ_count_seq=occ_count_seq,
                    occ_active_seq=occ_active_seq,
                    u_prev=u_prev,
                    ut_prev=ut_prev,
                    u_bar=u_bar,
                    ut_bar=ut_bar,
                    T_bar=T_now,
                )
            except Exception:
                # If QP fails for any reason, fall back to a safe control
                u0, ut0 = self.bnd.u_max, self.bnd.ut_min

            # ========== Step 6: Apply First Control (Receding Horizon) ==========
            # Fixed diffuser angle (kept outside QP to preserve convexity)
            phi0 = float(
                min(max(self.cfg.phi_nominal, self.bnd.phi_min), self.bnd.phi_max)
            )

            # store controls (on GPU tensor)
            control_vars[t, 0] = u0
            control_vars[t, 1] = ut0
            control_vars[t, 2] = phi0

            # Advance CFD simulation one step with the chosen action
            vel, tem = self.fwd.step(vel, tem, (float(u0), float(ut0), float(phi0)))

            u_prev = float(u0)
            ut_prev = float(ut0)

            # free GPU memory periodically (same style as your main loop)
            if (t + 1) % 10 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return control_vars
