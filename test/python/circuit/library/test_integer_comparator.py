# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of integer comparison circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import IntegerComparator, IntegerComparatorGate
from qiskit.synthesis.arithmetic import synth_integer_comparator_2s, synth_integer_comparator_greedy
from qiskit.quantum_info import Operator, Statevector
from test import QiskitTestCase


@ddt
class TestIntegerComparator(QiskitTestCase):
    """Test the integer comparator circuit."""

    def assertComparisonIsCorrect(self, comp, num_state_qubits, value, geq):
        """Assert that the comparator output is correct."""
        qc = QuantumCircuit(2 * num_state_qubits)  # initialize circuit
        qc.h(list(range(num_state_qubits)))  # set equal superposition state
        qc.append(comp, list(range(comp.num_qubits)))  # add comparator

        # run simulation
        tqc = transpile(qc)  # trigger the HLS if necessary
        statevector = Statevector(tqc)
        for i, amplitude in enumerate(statevector):
            prob = np.abs(amplitude) ** 2
            if prob > 1e-6:
                # equal superposition
                self.assertEqual(True, np.isclose(1.0, prob * 2.0**num_state_qubits))
                b_value = f"{i:b}".rjust(qc.width(), "0")
                x = int(b_value[(-num_state_qubits):], 2)
                comp_result = int(b_value[-num_state_qubits - 1], 2)
                if geq:
                    self.assertEqual(x >= value, comp_result == 1)
                else:
                    self.assertEqual(x < value, comp_result == 1)

    @data(
        [1, 0, True],
        [1, 1, True],
        [2, -1, True],
        [3, 5, True],
        [3, 2, True],
        [3, 2, False],
        [4, 6, False],
    )
    @unpack
    def test_fixed_value_comparator(self, num_state_qubits, value, geq):
        """Test the fixed value comparator circuit."""
        # build the circuit with the comparator
        for constructor, warn in [
            (IntegerComparator, True),
            (synth_integer_comparator_2s, False),
            (synth_integer_comparator_greedy, False),
        ]:
            with self.subTest(constructor=constructor):
                if warn:
                    with self.assertWarns(DeprecationWarning):
                        comp = constructor(num_state_qubits, value, geq=geq)
                else:
                    comp = constructor(num_state_qubits, value, geq=geq)

                self.assertComparisonIsCorrect(comp, num_state_qubits, value, geq)

    def test_mutability(self):
        """Test changing the arguments of the comparator."""

        with self.assertWarns(DeprecationWarning):
            comp = IntegerComparator()

        with self.subTest(msg="missing num state qubits and value"):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())

        comp.num_state_qubits = 2

        with self.subTest(msg="missing value"):
            with self.assertRaises(AttributeError):
                _ = str(comp.draw())

        comp.value = 0
        comp.geq = True

        with self.subTest(msg="updating num state qubits"):
            comp.num_state_qubits = 1
            self.assertComparisonIsCorrect(comp, 1, 0, True)

        with self.subTest(msg="updating the value"):
            comp.num_state_qubits = 3
            comp.value = 2
            self.assertComparisonIsCorrect(comp, 3, 2, True)

        with self.subTest(msg="updating geq"):
            comp.geq = False
            self.assertComparisonIsCorrect(comp, 3, 2, False)

    @data(1, 2, 3, 4)
    def test_greedy_operator(self, num_state_qubits):
        """Check both target states and phases, including boundary and fractional thresholds."""
        limit = 2**num_state_qubits
        columns = np.arange(2 * limit)
        values = [*range(-1, limit + 2), 0.1, 1.1, limit - 0.1]
        for value in values:
            for geq in (False, True):
                matches = columns % limit >= value
                if not geq:
                    matches = ~matches
                rows = columns ^ (matches.astype(int) * limit)
                expected = np.zeros((2 * limit, 2 * limit))
                expected[rows, columns] = 1
                for constructor in (synth_integer_comparator_greedy, IntegerComparatorGate):
                    with self.subTest(value=value, geq=geq, constructor=constructor):
                        circuit = constructor(num_state_qubits, value, geq)
                        self.assertEqual(Operator(circuit), Operator(expected))

    def test_greedy_numpy_width(self):
        """The synthesis function also accepts NumPy integer widths."""
        for value in (5, 8, 13):
            with self.subTest(value=value):
                self.assertEqual(
                    synth_integer_comparator_greedy(np.int64(4), value),
                    synth_integer_comparator_greedy(4, value),
                )

    @data(1, 4, 8, 16)
    def test_greedy_gate_count(self, num_state_qubits):
        """Comparison synthesis needs at most n controlled X gates and one target X."""
        limit = 2**num_state_qubits
        for value in (0, 1, limit // 2 - 1, limit // 2, limit // 2 + 1, limit - 1, limit):
            for geq in (False, True):
                with self.subTest(value=value, geq=geq):
                    circuit = synth_integer_comparator_greedy(num_state_qubits, value, geq)
                    self.assertEqual(circuit.num_qubits, num_state_qubits + 1)
                    self.assertLessEqual(circuit.size(), num_state_qubits + 1)

    @data((4, 5, 42), (5, 13, 100), (6, 32, 1))
    @unpack
    def test_greedy_default_hls_level3(self, num_state_qubits, value, cx_bound):
        """The default HLS retains the reduction at level 3 without clean auxiliary qubits."""
        circuit = QuantumCircuit(num_state_qubits + 1)
        circuit.append(IntegerComparatorGate(num_state_qubits, value), circuit.qubits)
        synthesized = transpile(
            circuit,
            basis_gates=["u", "cx"],
            optimization_level=3,
            qubits_initially_zero=False,
            seed_transpiler=42,
        )
        self.assertEqual(synthesized.num_qubits, circuit.num_qubits)
        self.assertLessEqual(synthesized.count_ops().get("cx", 0), cx_bound)
        self.assertEqual(Operator.from_circuit(synthesized), Operator(circuit))


if __name__ == "__main__":
    unittest.main()
