"""Tests for goad_v1.calibration and the error decomposition."""
import numpy as np
from ase import Atoms

from goad_v1 import calibration as cal
from goad_v1 import hybrid


# --- error decomposition (hybrid) ------------------------------------------
def test_error_decomposition_identity():
    # arbitrary numbers: total must equal eps_g + eps_v and the raw difference
    d = hybrid.error_decomposition(0.10, -0.30, -0.45)
    assert abs(d["eps_g"] - (0.10 - -0.30)) < 1e-12
    assert abs(d["eps_v"] - (-0.30 - -0.45)) < 1e-12
    assert abs(d["total"] - (0.10 - -0.45)) < 1e-12
    assert abs(d["total"] - (d["eps_g"] + d["eps_v"])) < 1e-12


def test_error_decomposition_pure_geometry():
    # model energy correct at the reference geometry (eps_v == 0) -> all error
    # is geometric (a detached explorer minimum).
    d = hybrid.error_decomposition(0.9, -0.5, -0.5)
    assert abs(d["eps_v"]) < 1e-12
    assert abs(d["eps_g"] - 1.4) < 1e-12


# --- heavy_atom_count ------------------------------------------------------
def test_heavy_atom_count_forms():
    assert cal.heavy_atom_count("C3H8O3") == 6          # glycerol
    assert cal.heavy_atom_count("CO2") == 3
    assert cal.heavy_atom_count(Atoms("C6H6")) == 6
    assert cal.heavy_atom_count(["C", "H", "H", "O"]) == 2


# --- isotonic PAVA + threshold fit -----------------------------------------
def test_pava_is_monotone_nondecreasing():
    y = [0.3, 0.1, 0.2, 0.9, 0.4]
    f = cal._pava(y, [1, 1, 1, 1, 1])
    assert np.all(np.diff(f) >= -1e-12)


def test_fit_threshold_recovers_known_crossing():
    # construct data whose mean |eps_v| crosses tau=0.2 at n=7
    rng = np.random.default_rng(0)
    tau, n_star = 0.2, 7.0
    n = rng.integers(1, 14, size=800)
    slope = (tau - 0.05) / n_star
    e = np.abs(0.05 + slope * n + rng.normal(0, 0.01, size=n.shape))
    fit = cal.fit_threshold(n, e, tau=tau)
    assert 6 <= fit.threshold <= 8            # recovers ~7
    assert abs(fit.crossing - n_star) <= 1.5


def test_fit_threshold_never_and_always_sentinels():
    n = np.arange(1, 11)
    lo = cal.fit_threshold(n, np.full(10, 0.01), tau=0.2)   # never reaches tau
    assert lo.threshold == 11 and lo.crossing == float("inf")
    hi = cal.fit_threshold(n, np.full(10, 0.5), tau=0.2)    # always above tau
    assert hi.threshold == 1 and hi.crossing == float("-inf")


# --- objective-aligned threshold (cost-at-target) --------------------------
def test_fit_threshold_for_target_meets_target_on_its_own_data():
    rng = np.random.default_rng(1)
    tau, n_star, target = 0.2, 7.0, 0.95
    n = rng.integers(1, 14, size=2000)
    slope = (tau - 0.05) / n_star
    e = np.abs(0.05 + slope * n + rng.normal(0, 0.05, size=n.shape))
    fit = cal.fit_threshold_for_target(n, e, tau=tau, target=target)
    # by construction it must reach the target on the calibration data
    assert fit.achieved_within_tau >= target
    # and it must be cheaper than "refine everything" (threshold > 1) because
    # the smallest molecules are already within tolerance
    assert fit.threshold > 1


def test_target_threshold_is_below_mean_crossing_under_scatter():
    # With noise, hitting a high within-tau target needs an EARLIER cutoff than
    # the mean-error crossing.
    rng = np.random.default_rng(2)
    tau, n_star = 0.2, 8.0
    n = rng.integers(1, 14, size=4000)
    slope = (tau - 0.05) / n_star
    e = np.abs(0.05 + slope * n + rng.normal(0, 0.06, size=n.shape))
    mean_fit = cal.fit_threshold(n, e, tau=tau)
    tgt_fit = cal.fit_threshold_for_target(n, e, tau=tau, target=0.95)
    assert tgt_fit.threshold <= mean_fit.crossing


def test_fit_threshold_for_target_never_refine_when_explorer_always_good():
    n = np.arange(1, 11)
    e = np.full(10, 0.01)                       # explorer within tau everywhere
    fit = cal.fit_threshold_for_target(n, e, tau=0.2, target=0.95)
    assert fit.threshold == 11                  # refine nothing by size


# --- policy evaluation -----------------------------------------------------
def _toy_systems():
    # magnetic systems: small (n=2) explorer-good, large (n=10) explorer-bad;
    # refiner always accurate. Plus a couple of non-magnetic systems.
    return [
        cal.System(2, True, e_explorer=0.05, e_refiner=0.05),
        cal.System(2, True, e_explorer=0.05, e_refiner=0.05),
        cal.System(10, True, e_explorer=0.60, e_refiner=0.05),
        cal.System(10, True, e_explorer=0.60, e_refiner=0.05),
        cal.System(3, False, e_explorer=0.99, e_refiner=0.04),
    ]


def test_non_magnetic_error_is_policy_invariant():
    sysx = cal.System(3, False, e_explorer=0.99, e_refiner=0.04)
    err_never, used_never = cal._system_error(sysx, cal.policy_never())
    err_always, used_always = cal._system_error(sysx, cal.policy_always())
    assert err_never == err_always == 0.04       # uses accurate model regardless
    assert used_never is False and used_always is False


def test_calibrated_beats_fixed_when_crossing_differs():
    systems = _toy_systems()
    # true crossing is between 2 and 10; a calibrated cutoff of ~6-10 refines
    # only the large ones. fixed@3 would waste refiner calls on small mols;
    # fixed@11 would under-refine the large ones.
    policies = {
        "never": cal.policy_never(),
        "always": cal.policy_always(),
        "fixed@11": cal.policy_fixed(11),     # too high: misses large
        "calibrated": cal.policy_fixed(6),    # correct: refines only large
    }
    res = cal.evaluate_policies(systems, policies)
    # calibrated should match 'always' accuracy on magnetics but cost less
    assert res["calibrated"].mae_magnetic <= res["fixed@11"].mae_magnetic
    assert abs(res["calibrated"].mae_magnetic - res["always"].mae_magnetic) < 1e-9
    assert res["calibrated"].refiner_calls < res["always"].refiner_calls
    # never-refine is worst on magnetic large molecules
    assert res["never"].mae_magnetic > res["calibrated"].mae_magnetic


def test_rank_policies_orders_by_accuracy_then_cost():
    systems = _toy_systems()
    res = cal.evaluate_policies(systems, {
        "always": cal.policy_always(),
        "calibrated": cal.policy_fixed(6),
        "never": cal.policy_never(),
    })
    ranked = cal.rank_policies(res)
    # calibrated ties always on accuracy but wins on cost -> ranked first
    assert ranked[0][0] == "calibrated"
    assert ranked[-1][0] == "never"


def test_within_tau_and_cost_at_target_ranking():
    systems = _toy_systems()   # 4 magnetic: two small (n=2), two large (n=10)
    tau = 0.2
    res = cal.evaluate_policies(systems, {
        "never": cal.policy_never(),        # large ones stay at 0.60 > tau
        "always": cal.policy_always(),      # all refined -> within tau
        "calibrated": cal.policy_fixed(6),  # refines only the large ones
    }, tau=tau)
    # within-tau: never misses the 2 large (2/4=0.5); always & calibrated = 1.0
    assert abs(res["never"].within_tau - 0.5) < 1e-9
    assert abs(res["always"].within_tau - 1.0) < 1e-9
    assert abs(res["calibrated"].within_tau - 1.0) < 1e-9
    # cost-at-target: calibrated reaches 100% within tau at 2 calls vs always's 4
    ranked = cal.rank_by_cost_at_target(res, target=0.95)
    assert ranked[0][0] == "calibrated"
    assert res["calibrated"].refiner_calls < res["always"].refiner_calls
    # never misses the target -> ranked last
    assert ranked[-1][0] == "never"

