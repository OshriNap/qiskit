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
from qiskit.transpiler import CouplingMap, PassManager, TranspilerError
from qiskit.transpiler.passes import (
    CheckMap,
    ConsolidateBlocks,
    SabreLayout,
    SabreSwap,
    UnitarySynthesis,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase

BASIS = ["cx", "rz", "sx", "x"]


def _qft(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.append(QFTGate(num_qubits), qc.qubits)
    return qc.decompose(reps=3)


def _routed_ok(circuit, coupling_map):
    check = CheckMap(coupling_map)
    check(circuit)
    return check.property_set["is_swap_mapped"]


def _resynthesized(circuit):
    """Two-qubit gate count and depth after two-qubit block resynthesis."""
    out = PassManager(
        [ConsolidateBlocks(basis_gates=BASIS), UnitarySynthesis(basis_gates=BASIS)]
    ).run(circuit)
    return out.num_nonlocal_gates(), out.depth(lambda inst: inst.operation.num_qubits == 2)


@ddt
class TestSabreTrialSelection(QiskitTestCase):
    """Trial selection: by fewest swaps (the default), or with ``trial_selection="two_qubit_cost"``
    by the estimated two-qubit cost of each routed trial after block resynthesis."""

    def test_default_is_fewest_swaps(self):
        """Without the option, both passes keep the fewest-swaps choice."""
        self.assertEqual(SabreSwap(CouplingMap.from_line(3)).trial_selection, "swaps")
        self.assertEqual(SabreLayout(CouplingMap.from_line(3)).trial_selection, "swaps")

    def test_invalid_selection(self):
        """An unknown selection is rejected."""
        with self.assertRaises(TranspilerError):
            SabreSwap(CouplingMap.from_line(3), trial_selection="depth")
        with self.assertRaises(TranspilerError):
            SabreLayout(CouplingMap.from_line(3), trial_selection="depth")

    def test_cost_selection_prefers_shallower_trial(self):
        """With 20 routing trials, the fewest-swaps trial is deeper after resynthesis than a trial
        with one more swap whose swaps merge into blocks; the cost-based choice picks the latter."""
        qc = QuantumCircuit(5)
        for a, b in [(2, 4), (4, 2), (1, 4), (1, 2), (4, 0), (3, 0)]:
            qc.cx(a, b)
        cmap = CouplingMap.from_line(5)

        def route(selection):
            routing = SabreSwap(
                cmap, heuristic="decay", seed=7, trials=20, trial_selection=selection
            )
            return PassManager([routing]).run(qc)

        by_swaps, by_cost = route("swaps"), route("two_qubit_cost")
        self.assertEqual(by_swaps.count_ops()["swap"], 5)
        self.assertEqual(by_cost.count_ops()["swap"], 6)
        self.assertEqual(_resynthesized(by_swaps), (19, 19))
        self.assertEqual(_resynthesized(by_cost), (16, 12))
        for routed in (by_swaps, by_cost):
            self.assertTrue(_routed_ok(routed, cmap))

    @data("swaps", "two_qubit_cost")
    def test_layout_deterministic_with_seed(self, selection):
        """The choice between layout trials is deterministic for a fixed seed."""
        qc = _qft(12)
        cmap = CouplingMap.from_heavy_hex(3)
        outputs = []
        for _ in range(3):
            layout_pass = SabreLayout(
                cmap, seed=2026, swap_trials=12, layout_trials=12, trial_selection=selection
            )
            out = PassManager([layout_pass]).run(qc)
            outputs.append((out, layout_pass.property_set["layout"]))
        for out, layout in outputs[1:]:
            self.assertEqual(out, outputs[0][0])
            self.assertEqual(layout, outputs[0][1])

    @data("swaps", "two_qubit_cost")
    def test_swap_deterministic_with_seed(self, selection):
        """The choice between routing trials is deterministic for a fixed seed."""
        qc = _qft(10)
        cmap = CouplingMap.from_line(10)
        outputs = [
            PassManager([SabreSwap(cmap, seed=11, trials=12, trial_selection=selection)]).run(qc)
            for _ in range(3)
        ]
        for out in outputs[1:]:
            self.assertEqual(out, outputs[0])
        self.assertTrue(_routed_ok(outputs[0], cmap))

    @data(1, 5, 16)
    def test_layout_and_routing_are_valid(self, trials):
        """Whatever trial is chosen, the routed circuit respects the coupling map."""
        cmap = CouplingMap.from_heavy_hex(3)
        for seed in range(3):
            layout_pass = SabreLayout(
                cmap,
                seed=seed,
                swap_trials=trials,
                layout_trials=trials,
                trial_selection="two_qubit_cost",
            )
            self.assertTrue(_routed_ok(PassManager([layout_pass]).run(_qft(14)), cmap))

    def test_with_opaque_gates_and_barriers(self):
        """Opaque two-qubit gates, three-qubit gates and barriers do not break the cost-based
        choice, and the routing stays valid."""
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
            layout_pass = SabreLayout(
                cmap, seed=seed, swap_trials=10, layout_trials=10, trial_selection="two_qubit_cost"
            )
            self.assertTrue(_routed_ok(PassManager([layout_pass]).run(qc), cmap))
            routing = SabreSwap(cmap, seed=seed, trials=10, trial_selection="two_qubit_cost")
            self.assertTrue(_routed_ok(PassManager([routing]).run(qc), cmap))

    @data(0, 1, 2, 3)
    def test_presets_choose_by_cost_only_at_levels_2_and_3(self, optimization_level):
        """Only the preset levels that resynthesize two-qubit blocks after routing choose by
        estimated two-qubit cost."""
        backend = GenericBackendV2(num_qubits=6, coupling_map=CouplingMap.from_line(6), seed=3)
        qc = QuantumCircuit(6)
        for i, j in ((0, 5), (1, 4), (2, 3), (0, 3), (5, 1), (2, 4)):
            qc.cx(i, j)
        selections = set()

        def callback(pass_, **_):
            if isinstance(pass_, (SabreLayout, SabreSwap)):
                selections.add(pass_.trial_selection)

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend, seed_transpiler=1
        )
        pm.run(qc, callback=callback)
        expected = {"two_qubit_cost"} if optimization_level >= 2 else {"swaps"}
        self.assertEqual(selections, expected)

    @data(1, 2, 3)
    def test_full_pipeline_is_exact(self, optimization_level):
        """The preset pipelines, with the trials they choose, still produce an equivalent circuit."""
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
