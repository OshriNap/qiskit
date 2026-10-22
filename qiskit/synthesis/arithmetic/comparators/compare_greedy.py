# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Integer comparator based on disjoint intervals of control states."""

import numpy as np

from qiskit.circuit import QuantumCircuit


def synth_integer_comparator_greedy(
    num_state_qubits: int, value: int, geq: bool = True
) -> QuantumCircuit:
    r"""Implement an integer comparison without auxiliary qubits.

    Partition the accepted values into disjoint, aligned power-of-two intervals.
    Each interval fixes a binary prefix and is implemented by one multi-controlled
    X gate on those prefix bits. This uses at most ``num_state_qubits`` controlled
    X gates, with an optional X gate to complement the comparison.

    Args:
        num_state_qubits: The number of qubits encoding the value to compare to.
        value: The value to compare to.
        geq: If ``True`` flip the target bit if the qubit state is :math:`\geq` than the value,
            otherwise implement :math:`<`.

    Returns:
        A circuit implementing the integer comparator.

    """
    circuit = QuantumCircuit(num_state_qubits + 1)

    if value <= 0:  # condition always satisfied for non-positive values
        if geq:  # otherwise the condition is never satisfied
            circuit.x(num_state_qubits)

        return circuit

    # Choose the comparison with fewer accepted values, complementing the target if needed.
    value = int(np.ceil(value))
    if (value < 2 ** (num_state_qubits - 1) and geq) or (
        value > 2 ** (num_state_qubits - 1) and not geq
    ):
        geq = not geq
        circuit.x(num_state_qubits)

    lower, upper = (value, 2**num_state_qubits) if geq else (0, value)
    while lower < upper:
        # An aligned power-of-two interval fixes only its high-order bits.
        free_bits = (lower & -lower).bit_length() - 1 if lower else num_state_qubits
        free_bits = min(free_bits, int(upper - lower).bit_length() - 1)
        circuit.mcx(
            list(range(free_bits, num_state_qubits)),
            num_state_qubits,
            ctrl_state=lower >> free_bits,
        )
        lower += 1 << free_bits

    return circuit
