"""Check the production initializer independently of the optimizer."""

import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from src.stage2_optimize import (
    build_dicke_10_6_bartschi_eidenbenz_circuit,
    build_uniform_dicke_statevector,
    prove_all_weight6_subsets_feasible,
)


class DickePreparationTests(unittest.TestCase):
    def test_uniform_amplitudes_and_phase(self):
        actual = Statevector.from_instruction(
            build_dicke_10_6_bartschi_eidenbenz_circuit()
        ).data
        expected = build_uniform_dicke_statevector(10, 6)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_capacity_guard_rejects_binding_case(self):
        proof = prove_all_weight6_subsets_feasible(
            costs=np.ones(10),
            capacities=np.full(10, 5.0),
            k=6,
            budget_limit_c=25,
            capacity_limit_h=22,
        )
        self.assertFalse(proof["all_weight_k_subsets_proved_feasible"])


if __name__ == "__main__":
    unittest.main()
