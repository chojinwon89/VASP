import numpy as np
from ase import Atoms

from goad_v1.ga import genetic_algorithm as ga_module
from goad_v1.ga.genetic_algorithm import GeneticAlgorithm


def _make_test_ga(n_workers=1, calculator_type=None):
    surface = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=[True, True, False],
    )
    molecule = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    return GeneticAlgorithm(
        surface=surface,
        molecule=molecule,
        calculator=object(),
        calculator_type=calculator_type,
        surface_energy=0.0,
        molecule_energy=0.0,
        generations=1,
        population_size=2,
        verbose=False,
        n_workers=n_workers,
    )


def _individual(x):
    return {
        "position": np.array([x, 0.0, 2.0]),
        "orientation": np.zeros(3),
        "torsions": np.zeros(0),
        "energy": None,
        "structure": None,
    }


def test_evaluate_population_serial_fallback_updates_best_and_history():
    ga = _make_test_ga(n_workers=1, calculator_type="sevennet_omni")
    ga.population = [_individual(2.0), _individual(1.0)]
    ga._calculate_energy = lambda individual: float(individual["position"][0])  # noqa: SLF001

    ga._evaluate_population()

    assert [ind["energy"] for ind in ga.population] == [2.0, 1.0]
    assert ga.fitness_history == [2.0, 1.0]
    assert ga.best_energy == 1.0
    assert np.allclose(ga.best_individual["position"], [1.0, 0.0, 2.0])


def test_evaluate_population_parallel_path_collects_results(monkeypatch):
    ga = _make_test_ga(n_workers=3, calculator_type="sevennet_omni")
    ga.population = [_individual(0.5), _individual(1.5)]

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self.max_workers = kwargs["max_workers"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, pending):
            assert self.max_workers == 3
            pending = list(pending)
            assert [idx for idx, _ in pending] == [0, 1]
            return iter([(0, 4.2, "s0"), (1, -0.8, "s1")])

    monkeypatch.setattr(ga_module, "ProcessPoolExecutor", FakeExecutor)

    ga._evaluate_population()

    assert [ind["energy"] for ind in ga.population] == [4.2, -0.8]
    assert [ind["structure"] for ind in ga.population] == ["s0", "s1"]
    assert ga.fitness_history == [4.2, -0.8]
    assert ga.best_energy == -0.8
    assert np.allclose(ga.best_individual["position"], [1.5, 0.0, 2.0])


def test_evaluate_population_parallel_failure_falls_back_to_serial(monkeypatch):
    ga = _make_test_ga(n_workers=2, calculator_type="sevennet_omni")
    ga.population = [_individual(3.0), _individual(1.0)]
    ga._calculate_energy = lambda individual: float(individual["position"][0]) - 10.0  # noqa: SLF001

    class BrokenExecutor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(ga_module, "ProcessPoolExecutor", BrokenExecutor)

    ga._evaluate_population()

    assert [ind["energy"] for ind in ga.population] == [-7.0, -9.0]
    assert ga.fitness_history == [-7.0, -9.0]
    assert ga.best_energy == -9.0
