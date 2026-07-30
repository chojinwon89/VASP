"""Data-driven calibration of the explorer/refiner size threshold.

Motivation
----------
``goad_v1.hybrid`` routes *large* molecules on magnetic surfaces to an accurate
refiner because CHGNet's binding-energy VALUE error ``eps_v`` (see
``hybrid.error_decomposition``) grows with molecule size. The size cutoff was a
hard-coded default of 6 heavy atoms. This module replaces that guess with a
threshold *fitted from measured* ``(n_heavy, |eps_v|)`` data, and provides a
harness to test whether the fitted threshold actually beats the fixed one.

Two composable pieces
---------------------
1. ``fit_threshold`` -- monotone (isotonic / pool-adjacent-violators) fit of
   ``|eps_v|`` versus heavy-atom count, returning the crossing ``n*`` where the
   value error first exceeds a tolerance ``tau``. This is the calibrated cutoff.
2. ``evaluate_policies`` -- score competing routing policies (never-refine,
   always-refine, fixed@k, calibrated) on a workload, reporting mean binding-
   energy error AND refiner-call cost so the accuracy/cost trade-off is explicit.

Only numpy is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms

# Default binding-energy tolerance (eV): |eps_v| above this is "inaccurate".
# ~0.1-0.2 eV is chemical-accuracy-ish for adsorption energetics.
DEFAULT_TAU = 0.2


# ---------------------------------------------------------------------------
# molecule size
# ---------------------------------------------------------------------------
def heavy_atom_count(mol: Union[Atoms, Sequence[str], str]) -> int:
    """Count non-hydrogen atoms in an ASE Atoms, a symbol list, or a formula.

    A formula string is parsed loosely: element symbols with optional counts,
    e.g. ``"C3H8O3"`` -> 6 heavy atoms.
    """
    if isinstance(mol, Atoms):
        return sum(1 for s in mol.get_chemical_symbols() if s != "H")
    if isinstance(mol, str):
        import re
        n = 0
        for sym, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", mol):
            if not sym:
                continue
            if sym == "H":
                continue
            n += int(cnt) if cnt else 1
        return n
    return sum(1 for s in mol if s != "H")


# ---------------------------------------------------------------------------
# isotonic (monotone non-decreasing) fit via pool-adjacent-violators
# ---------------------------------------------------------------------------
def _pava(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression. Returns per-point fit."""
    blocks: List[List[float]] = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([float(v), float(w), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            nw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / nw, nw, int(c1 + c2)])
    out: List[float] = []
    for v, _w, c in blocks:
        out.extend([v] * c)
    return np.asarray(out, dtype=float)


@dataclass
class ThresholdFit:
    """Result of ``fit_threshold``."""
    threshold: int                    # integer routing cutoff: refine if n_heavy >= threshold
    crossing: float                   # fractional n where fitted |eps_v| == tau
    tau: float
    grid_n: np.ndarray                # integer heavy-atom grid
    grid_error: np.ndarray            # fitted monotone |eps_v| on the grid
    bin_n: np.ndarray                 # observed heavy-atom bins
    bin_error: np.ndarray            # isotonic-fitted |eps_v| per bin
    bin_count: np.ndarray             # #observations per bin
    n_samples: int

    def predicted_error(self, n_heavy: int) -> float:
        """Fitted |eps_v| at a heavy-atom count (clamped to the grid range)."""
        return float(np.interp(n_heavy, self.grid_n, self.grid_error))


def fit_threshold(n_heavy: Sequence[int],
                  abs_eps_v: Sequence[float],
                  tau: float = DEFAULT_TAU) -> ThresholdFit:
    """Fit the calibrated size threshold from measured value errors.

    Bins ``|eps_v|`` by integer heavy-atom count, fits a non-decreasing curve
    (physical prior: value error grows with size), and returns the smallest
    integer ``n`` where the fitted error reaches ``tau``.

    Sentinels: if the curve never reaches ``tau`` the threshold is
    ``max(n)+1`` (crossing ``+inf``) -- never route by size; if it already
    exceeds ``tau`` at the smallest ``n`` the threshold is that ``n``
    (crossing ``-inf``) -- always route.
    """
    n = np.asarray(n_heavy, dtype=int)
    e = np.asarray(abs_eps_v, dtype=float)
    if n.size == 0:
        raise ValueError("no samples to fit")
    order = np.argsort(n)
    n, e = n[order], e[order]

    uniq = np.unique(n)
    means = np.array([e[n == u].mean() for u in uniq])
    counts = np.array([int((n == u).sum()) for u in uniq])
    fitted_bins = _pava(means, counts)

    n_min, n_max = int(uniq.min()), int(uniq.max())
    grid_n = np.arange(n_min, n_max + 1)
    grid_error = np.interp(grid_n, uniq, fitted_bins)

    if fitted_bins[-1] < tau:                       # never reaches tau
        threshold, crossing = n_max + 1, float("inf")
    elif fitted_bins[0] >= tau:                     # already above at smallest n
        threshold, crossing = n_min, float("-inf")
    else:
        # fractional crossing by linear interpolation on the monotone grid
        above = np.where(grid_error >= tau)[0][0]
        if above == 0:
            crossing = float(grid_n[0])
        else:
            n0, n1 = grid_n[above - 1], grid_n[above]
            e0, e1 = grid_error[above - 1], grid_error[above]
            crossing = float(n0 + (tau - e0) * (n1 - n0) / (e1 - e0)) \
                if e1 != e0 else float(n1)
        threshold = int(np.ceil(crossing - 1e-9))

    return ThresholdFit(threshold=threshold, crossing=crossing, tau=tau,
                        grid_n=grid_n, grid_error=grid_error,
                        bin_n=uniq, bin_error=fitted_bins, bin_count=counts,
                        n_samples=int(n.size))


@dataclass
class TargetThresholdFit:
    """Result of ``fit_threshold_for_target`` (objective-aligned calibration)."""
    threshold: int                 # refine if n_heavy >= threshold
    target: float                  # required within-tau rate
    tau: float
    achieved_within_tau: float     # within-tau rate on calibration data at threshold
    refine_fraction: float         # fraction of calibration samples refined at threshold
    n_samples: int


def fit_threshold_for_target(n_heavy: Sequence[int],
                             abs_eps_v: Sequence[float],
                             tau: float = DEFAULT_TAU,
                             target: float = 0.95) -> TargetThresholdFit:
    """Calibrate the size threshold to the *decision objective* directly.

    The mean-error crossing (:func:`fit_threshold`) answers "where does the
    average value error reach ``tau``", but routing at that point still leaves
    ~half of the borderline (noisy) systems above ``tau``, so it under-shoots a
    high within-tolerance target. This function instead picks the **largest**
    (cheapest) integer threshold ``k`` for which routing ``refine if n>=k``
    keeps at least ``target`` of the calibration samples within ``tau`` -- i.e.
    it minimises refiner cost subject to meeting the accuracy target, which is
    exactly what the threshold is for.

    A system is within tolerance if it is refined (``n>=k`` -> accurate model)
    or its explorer error is already ``<= tau``. ``within_tau(k)`` is monotone
    (refining more only helps), so the largest ``k`` meeting the target is the
    cost-optimal cutoff.
    """
    n = np.asarray(n_heavy, dtype=int)
    e = np.asarray(abs_eps_v, dtype=float)
    if n.size == 0:
        raise ValueError("no samples to fit")
    n_max = int(n.max())
    best_k = 1
    for k in range(n_max + 1, 0, -1):        # scan from "refine none" downward
        within = float(np.mean((n >= k) | (e <= tau)))
        if within >= target:
            best_k = k
            break
    refined = n >= best_k
    within = float(np.mean(refined | (e <= tau)))
    return TargetThresholdFit(threshold=best_k, target=target, tau=tau,
                              achieved_within_tau=within,
                              refine_fraction=float(np.mean(refined)),
                              n_samples=int(n.size))


# ---------------------------------------------------------------------------
# routing policies + head-to-head evaluation
# ---------------------------------------------------------------------------
@dataclass
class System:
    """One benchmark system with its (true) per-model value errors.

    ``e_explorer`` / ``e_refiner`` are the |binding-energy errors| of the
    spin-aware explorer (CHGNet) and the accurate refiner respectively. On a
    non-magnetic surface a single accurate model is used, so the reported error
    is ``e_refiner`` regardless of policy.
    """
    n_heavy: int
    magnetic: bool
    e_explorer: float
    e_refiner: float


# A policy maps (n_heavy, magnetic) -> refine? (only consulted for magnetic).
Policy = Callable[[int, bool], bool]


def policy_never() -> Policy:
    return lambda n, mag: False


def policy_always() -> Policy:
    return lambda n, mag: True


def policy_fixed(threshold: int) -> Policy:
    return lambda n, mag: n >= threshold


def policy_calibrated(fit: "Union[ThresholdFit, TargetThresholdFit]") -> Policy:
    thr = fit.threshold
    return lambda n, mag: n >= thr


def _system_error(sys: System, policy: Policy) -> Tuple[float, bool]:
    """Return (reported |error|, refiner_used) for one system under a policy."""
    if not sys.magnetic:
        return sys.e_refiner, False          # single accurate model; policy-invariant
    refine = policy(sys.n_heavy, sys.magnetic)
    return (sys.e_refiner if refine else sys.e_explorer), bool(refine)


@dataclass
class PolicyResult:
    name: str
    mae_all: float                 # mean |error| over all systems
    mae_magnetic: float            # mean |error| over magnetic systems only
    refiner_calls: int             # extra accurate-model relaxations spent
    n_systems: int
    n_magnetic: int
    within_tau: Optional[float] = None   # fraction of magnetic systems with |err| <= tau


def evaluate_policies(systems: Sequence[System],
                      policies: Dict[str, Policy],
                      tau: Optional[float] = None) -> Dict[str, PolicyResult]:
    """Score each named policy on the workload.

    Reports mean binding-energy error (accuracy) AND refiner-call count (cost).
    When ``tau`` is given, also reports ``within_tau`` -- the fraction of
    magnetic systems whose reported error is within tolerance -- which is the
    quantity the size threshold actually controls (small molecules the explorer
    already handles do not need an expensive refiner call).
    """
    mags = [s for s in systems if s.magnetic]
    results: Dict[str, PolicyResult] = {}
    for name, pol in policies.items():
        errs, mag_errs, calls, mag_within = [], [], 0, 0
        for s in systems:
            err, used = _system_error(s, pol)
            errs.append(err)
            if s.magnetic:
                mag_errs.append(err)
                if tau is not None and err <= tau:
                    mag_within += 1
            if used:
                calls += 1
        results[name] = PolicyResult(
            name=name,
            mae_all=float(np.mean(errs)) if errs else 0.0,
            mae_magnetic=float(np.mean(mag_errs)) if mag_errs else 0.0,
            refiner_calls=calls,
            n_systems=len(systems),
            n_magnetic=len(mags),
            within_tau=(mag_within / len(mags)) if (tau is not None and mags)
            else None,
        )
    return results


def rank_policies(results: Dict[str, PolicyResult]
                  ) -> List[Tuple[str, PolicyResult]]:
    """Rank by accuracy (lower magnetic MAE), tie-broken by lower cost.

    NOTE: pure-accuracy ranking trivially favours refining everything, because
    the refiner is accurate at every size and this metric ignores its cost. For
    the threshold decision use :func:`rank_by_cost_at_target` instead.
    """
    return sorted(results.items(),
                  key=lambda kv: (round(kv[1].mae_magnetic, 6),
                                  kv[1].refiner_calls))


def rank_by_cost_at_target(results: Dict[str, PolicyResult],
                           target: float = 0.95
                           ) -> List[Tuple[str, PolicyResult]]:
    """Rank by the threshold's real objective: cheapest policy that hits target.

    A refiner call is expensive (an accurate MLIP relaxation, or DFT), so the
    right question is not "who is most accurate" but "who reaches the accuracy
    target (``within_tau >= target``) at the lowest refiner cost". Policies that
    meet the target are ranked first by ascending cost; those that miss it are
    ranked afterwards by descending within-tolerance rate.
    """
    items = list(results.items())
    met = [(k, v) for k, v in items
           if v.within_tau is not None and v.within_tau >= target]
    unmet = [(k, v) for k, v in items
             if not (v.within_tau is not None and v.within_tau >= target)]
    met.sort(key=lambda kv: kv[1].refiner_calls)
    unmet.sort(key=lambda kv: -(kv[1].within_tau or 0.0))
    return met + unmet
