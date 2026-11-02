# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of how SabreLayout and SabreSwap choose between their trials."""

from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap, SabreLayout, SabreSwap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase


def _qft(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.append(QFTGate(num_qubits), qc.qubits)
    return qc.decompose(reps=3)


def _routed_ok(circuit, coupling_map):
    check = CheckMap(coupling_map)
    check(circuit)
    return check.property_set["is_swap_mapped"]


@ddt
class TestSabreTrialSelection(QiskitTestCase):
    """Trial selection: by fewest swaps with fewer than 10 trials, by the estimated two-qubit cost of
    each routed trial with 10 or more."""

    @data(4, 12)
    def test_layout_deterministic_with_seed(self, trials):
        """The choice between layout trials is deterministic for a fixed seed."""
        qc = _qft(12)
        cmap = CouplingMap.from_heavy_hex(3)
        outputs = []
        for _ in range(3):
            layout_pass = SabreLayout(cmap, seed=2026, swap_trials=trials, layout_trials=trials)
            out = PassManager([layout_pass]).run(qc)
            outputs.append((out, layout_pass.property_set["layout"]))
        for out, layout in outputs[1:]:
            self.assertEqual(out, outputs[0][0])
            self.assertEqual(layout, outputs[0][1])

    @data(4, 12)
    def test_swap_deterministic_with_seed(self, trials):
        """The choice between routing trials is deterministic for a fixed seed."""
        qc = _qft(10)
        cmap = CouplingMap.from_line(10)
        outputs = [
            PassManager([SabreSwap(cmap, seed=11, trials=trials)]).run(qc) for _ in range(3)
        ]
        for out in outputs[1:]:
            self.assertEqual(out, outputs[0])
        self.assertTrue(_routed_ok(outputs[0], cmap))

    @data(5, 10, 16)
    def test_layout_and_routing_are_valid(self, trials):
        """Whatever trial is chosen, the routed circuit respects the coupling map."""
        cmap = CouplingMap.from_heavy_hex(3)
        for seed in range(3):
            out = PassManager(
                [SabreLayout(cmap, seed=seed, swap_trials=trials, layout_trials=trials)]
            ).run(_qft(14))
            self.assertTrue(_routed_ok(out, cmap))

    @data(4, 10)
    def test_with_opaque_gates_and_barriers(self, trials):
        """Opaque two-qubit gates, three-qubit gates and barriers are handled by the trial cost
        estimate (and the routing stays valid)."""
        qc = QuantumCircuit(6)
        custom = Gate("custom_2q", 2, [])
        for i in range(5):
            qc.h(i)
            qc.append(custom, [i, (i + 3) % 6])
        qc.barrier()
        qc.ccx(0, 3, 5)
        qc.cx(5, 1)
        qc.swap(2, 4)
        qc.rzz(0.3, 0, 4)
        cmap = CouplingMap.from_line(6)
        for seed in range(3):
            out = PassManager(
                [SabreLayout(cmap, seed=seed, swap_trials=trials, layout_trials=trials)]
            ).run(qc)
            self.assertTrue(_routed_ok(out, cmap))
            routed = PassManager([SabreSwap(cmap, seed=seed, trials=trials)]).run(qc)
            self.assertTrue(_routed_ok(routed, cmap))

    @data(1, 2, 3)
    def test_full_pipeline_is_exact(self, optimization_level):
        """The preset pipelines, with the trials they choose, still produce an equivalent circuit
        (level 1 runs 5 Sabre trials, levels 2 and 3 run 20)."""
        backend = GenericBackendV2(num_qubits=6, coupling_map=CouplingMap.from_line(6), seed=3)
        qc = QuantumCircuit(6)
        for i in range(6):
            qc.h(i)
        for i, j in ((0, 5), (1, 4), (2, 3), (0, 3), (5, 1), (2, 4), (3, 0), (4, 5)):
            qc.cx(i, j)
            qc.rz(0.1 * (i + j + 1), j)
        for seed in range(4):
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level, backend=backend, seed_transpiler=seed
            )
            out = pm.run(qc)
            self.assertTrue(Operator.from_circuit(out).equiv(Operator(qc)))
