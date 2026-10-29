// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use fixedbitset::FixedBitSet;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::algo::toposort;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::{EdgeRef, NodeIndexable};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, pyfunction, wrap_pyfunction};

use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_util::IndexMap;
use smallvec::{SmallVec, smallvec};

use super::analyze_commutations;
use crate::commutation_checker::{CommutationChecker, CommutationError};
use approx::abs_diff_eq;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError, NodeType, PyDAGCircuit, Wire};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_synthesis::QiskitError;

/// Possible errors during the `CommutationCancellation` pass
#[derive(Debug, thiserror::Error)]
pub enum CommutationCancelError {
    #[error("Rotational gate with parameter expression encountered in cancellation {0:?}")]
    RotationalParameter(PackedOperation),
    #[error("Angle for operation {0} is not defined")]
    OperationUndefinedAngle(String),
    #[error(transparent)]
    DAGCircuit(#[from] DAGError),
    #[error(transparent)]
    Analysis(#[from] CommutationError),
}

impl From<CommutationCancelError> for PyErr {
    fn from(value: CommutationCancelError) -> Self {
        match value {
            CommutationCancelError::RotationalParameter(_) => {
                QiskitError::new_err(value.to_string())
            }
            CommutationCancelError::OperationUndefinedAngle(_) => {
                PyRuntimeError::new_err(value.to_string())
            }
            CommutationCancelError::Analysis(commutation_analysis_error) => {
                commutation_analysis_error.into()
            }
            CommutationCancelError::DAGCircuit(dagcircuit_inner_error) => {
                dagcircuit_inner_error.into()
            }
        }
    }
}

const _CUTOFF_PRECISION: f64 = 1e-5;
static ROTATION_GATES: [&str; 4] = ["p", "u1", "rz", "rx"];

static VAR_Z_MAP: [(&str, StandardGate); 3] = [
    ("rz", StandardGate::RZ),
    ("p", StandardGate::Phase),
    ("u1", StandardGate::U1),
];
static Z_ROTATIONS: [StandardGate; 8] = [
    StandardGate::Phase,
    StandardGate::Z,
    StandardGate::U1,
    StandardGate::RZ,
    StandardGate::T,
    StandardGate::Tdg,
    StandardGate::S,
    StandardGate::Sdg,
];
static X_ROTATIONS: [StandardGate; 4] = [
    StandardGate::X,
    StandardGate::RX,
    StandardGate::SX,
    StandardGate::SXdg,
];
static SUPPORTED_GATES: [StandardGate; 5] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::H,
    StandardGate::Y,
];

#[derive(Hash, Eq, PartialEq, Debug)]
enum GateOrRotation {
    Gate(StandardGate),
    ZRotation,
    XRotation,
}
#[derive(Hash, Eq, PartialEq, Debug)]
struct CancellationSetKey {
    gate: GateOrRotation,
    qubits: SmallVec<[Qubit; 2]>,
    com_set_index: usize,
    second_index: Option<usize>,
}

#[pyfunction(name = "cancel_commutations")]
#[pyo3(signature = (dag, commutation_checker, basis_gates=None, approximation_degree=1., *, cancel_cx5=false, reorder_for_depth=false))]
fn py_cancel_commutations(
    py: Python<'_>,
    dag: &mut PyDAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<Vec<String>>,
    approximation_degree: f64,
    cancel_cx5: bool,
    reorder_for_depth: bool,
) -> PyResult<()> {
    // Release the GIL, otherwise rayon workers deadlock when they need it for Python-defined gates.
    let dag = dag.try_write()?;
    py.detach(|| -> Result<(), CommutationCancelError> {
        cancel_commutations(dag, commutation_checker, basis_gates, approximation_degree)?;
        if cancel_cx5 {
            cancel_five_cx(dag);
        }
        if reorder_for_depth {
            reduce_two_qubit_depth(dag)?;
        }
        Ok(())
    })
    .map_err(Into::into)
}

/// Delete the endpoints of a five-CX identity, without moving the middle gates.
/// For distinct a, b, c: ABADB = BAD, where A=CX(a,b), B=CX(b,a),
/// D=CX(c,a). This follows from ABA=BAB and BD=DB. Also recognize the
/// reverse sequence and simultaneous reversal of every CX direction.
fn cancel_five_cx(dag: &mut DAGCircuit) {
    for (first, last) in find_five_cx_cancellations(dag) {
        dag.remove_op_node(first);
        dag.remove_op_node(last);
    }
}

/// Find disjoint endpoint pairs without mutating the input DAG.
pub(super) fn find_five_cx_cancellations(dag: &DAGCircuit) -> Vec<(NodeIndex, NodeIndex)> {
    if dag.get_op_counts().get("cx").copied().unwrap_or(0) < 5
        || dag.has_control_flow()
        || dag.width() != dag.num_qubits() + dag.num_clbits()
    {
        return Vec::new();
    }
    let mut used = FixedBitSet::new();
    let mut removals = Vec::new();
    for index in 0..dag.num_qubits() {
        let common = Qubit::new(index);
        let mut nodes = [NodeIndex::new(0); 5];
        let mut gates = [[common; 2]; 5];
        let mut len = 0;
        for node in dag.nodes_on_wire(Wire::Qubit(common)) {
            let NodeType::Operation(inst) = &dag[node] else {
                len = 0;
                continue;
            };
            if used.contains(node.index()) || inst.op.try_standard_gate() != Some(StandardGate::CX)
            {
                len = 0;
                continue;
            }
            if len == 5 {
                nodes.rotate_left(1);
                gates.rotate_left(1);
                len = 4;
            }
            nodes[len] = node;
            gates[len] = dag.get_qargs(inst.qubits).try_into().unwrap();
            len += 1;
            if len != 5 || !five_cx_identity(&gates) {
                continue;
            }
            // The common wire is consecutive by construction. The other repeated
            // wire must be consecutive too; the third wire occurs only once.
            let other = if gates[0][0] == common {
                gates[0][1]
            } else {
                gates[0][0]
            };
            let mut previous = None;
            let mut consecutive = true;
            for (&node, gate) in nodes.iter().zip(&gates) {
                if !gate.contains(&other) {
                    continue;
                }
                if let Some(prev) = previous {
                    let next = dag
                        .dag()
                        .edges_directed(prev, Direction::Outgoing)
                        .find(|edge| *edge.weight() == Wire::Qubit(other))
                        .map(|edge| edge.target());
                    if next != Some(node) {
                        consecutive = false;
                        break;
                    }
                }
                previous = Some(node);
            }
            if !consecutive {
                continue;
            }
            // Match against the original graph and reserve the whole window,
            // including retained nodes. Stable graph indices can contain holes.
            if used.is_empty() {
                used.grow(dag.dag().node_bound());
            }
            for node in nodes {
                used.insert(node.index());
            }
            removals.push((nodes[0], nodes[4]));
            len = 0;
        }
    }
    removals
}

fn five_cx_identity(g: &[[Qubit; 2]; 5]) -> bool {
    fn forward(g: &[[Qubit; 2]; 5]) -> bool {
        let [a, b] = g[0];
        g[2] == [a, b]
            && g[1] == [b, a]
            && g[4] == [b, a]
            && ((g[3][1] == a && g[3][0] != a && g[3][0] != b)
                || (g[3][0] == b && g[3][1] != a && g[3][1] != b))
    }
    forward(g) || forward(&[g[4], g[3], g[2], g[1], g[0]])
}

/// Advance adjacent, exactly commuting CX/CZ gates without delaying any wire.
/// For A(u,s) then B(s,v), swap only if input_depth(u) > max(input_depth(s),
/// input_depth(v)). B finishes earlier, while A's finish time stays unchanged.
/// One forward scan permits at most one interchange per visited operation.
fn reduce_two_qubit_depth(dag: &mut DAGCircuit) -> Result<(), DAGError> {
    if dag.has_control_flow()
        || !dag.vars().is_empty()
        || dag.get_op_counts().get("cx").copied().unwrap_or(0)
            + dag.get_op_counts().get("cz").copied().unwrap_or(0)
            < 2
    {
        return Ok(());
    }
    let num_qubits = dag.num_qubits();
    let mut arrival = vec![0usize; num_qubits + dag.num_clbits()];
    let mut tails = vec![None; num_qubits];
    let mut before = vec![[0usize; 2]; dag.dag().node_bound()];
    let mut ends = vec![0usize; dag.dag().node_bound()];
    // Independent operations need no lexicographic ordering for the arrival times.
    let order = toposort(dag.dag(), None).expect("DAGCircuit can't have a cycle");
    for node in order {
        let NodeType::Operation(inst) = &dag[node] else {
            continue;
        };
        let qargs = dag.get_qargs(inst.qubits);
        let cargs = dag.get_cargs(inst.clbits);
        let gate = inst.op.try_standard_gate();
        let pair = matches!(gate, Some(StandardGate::CX | StandardGate::CZ)).then(|| {
            let qubits = [qargs[0], qargs[1]];
            let inputs = [arrival[qubits[0].index()], arrival[qubits[1].index()]];
            let previous = [tails[qubits[0].index()], tails[qubits[1].index()]];
            (qubits, inputs, previous)
        });
        let finish = qargs
            .iter()
            .map(|q| arrival[q.index()])
            .chain(cargs.iter().map(|c| arrival[num_qubits + c.index()]))
            .max()
            .unwrap_or(0)
            + usize::from(qargs.len() == 2 && !inst.op.directive());
        ends[node.index()] = finish;
        for qubit in qargs {
            arrival[qubit.index()] = finish;
            tails[qubit.index()] = Some(node);
        }
        for clbit in cargs {
            arrival[num_qubits + clbit.index()] = finish;
        }
        let Some((qubits, inputs, previous)) = pair else {
            continue;
        };
        before[node.index()] = inputs;
        for shared in 0..2 {
            let Some(first) = previous[shared] else {
                continue;
            };
            let first_inst = dag[first].unwrap_operation();
            let first_gate = first_inst.op.try_standard_gate();
            if !matches!(first_gate, Some(StandardGate::CX | StandardGate::CZ)) {
                continue;
            }
            let args = dag.get_qargs(first_inst.qubits);
            let common = qubits[shared];
            let first_shared = usize::from(args[1] == common);
            if args[1 - first_shared] == qubits[1 - shared] {
                continue;
            }
            let commute = match (first_gate.unwrap(), gate.unwrap()) {
                (StandardGate::CZ, StandardGate::CZ) => true,
                (StandardGate::CX, StandardGate::CX) => first_shared == shared,
                (StandardGate::CX, StandardGate::CZ) => first_shared == 0,
                (StandardGate::CZ, StandardGate::CX) => shared == 0,
                _ => unreachable!(),
            };
            let x = before[first.index()][1 - first_shared];
            let y = before[first.index()][first_shared];
            let z = inputs[1 - shared];
            if !commute || x <= y.max(z) {
                continue;
            }
            dag.swap_nodes(first, node)?;
            let earlier_finish = y.max(z) + 1;
            ends[node.index()] = earlier_finish;
            before[node.index()][shared] = y;
            before[first.index()][first_shared] = earlier_finish;
            arrival[common.index()] = ends[first.index()];
            tails[common.index()] = Some(first);
            arrival[qubits[1 - shared].index()] = earlier_finish;
            tails[qubits[1 - shared].index()] = Some(node);
            break;
        }
    }
    Ok(())
}

pub fn cancel_commutations(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<Vec<String>>,
    approximation_degree: f64,
) -> Result<(), CommutationCancelError> {
    let basis = basis_gates.unwrap_or_default();
    let z_var_gate = dag
        .get_op_counts()
        .keys()
        .find_map(|g| {
            VAR_Z_MAP
                .iter()
                .find(|(key, _)| *key == g.as_str())
                .map(|(_, gate)| gate)
        })
        .or_else(|| {
            // Fallback to the first matching key from basis if there is no match in dag.op_names
            basis.iter().find_map(|g| {
                VAR_Z_MAP
                    .iter()
                    .find(|(key, _)| *key == g.as_str())
                    .map(|(_, gate)| gate)
            })
        });
    let sx_supported = dag.get_op_counts().contains_key("sx") || basis.iter().any(|x| x == "sx");
    let x_supported = dag.get_op_counts().contains_key("x") || basis.iter().any(|x| x == "x");

    // RZ and P/U1 have a phase difference of angle/2, which we need to account for
    let z_phase_shift = match z_var_gate {
        Some(z_var_gate) => {
            if z_var_gate == &StandardGate::RZ {
                |gate_name: &str, angle: f64| -> f64 {
                    if ["u1", "p"].contains(&gate_name) {
                        angle / 2.
                    } else {
                        0.
                    }
                }
            } else {
                |gate_name: &str, angle: f64| -> f64 {
                    if gate_name == "rz" { -angle / 2. } else { 0. }
                }
            }
        }
        // if there's no z_var_gate detected, Z rotations are not merged, so we have no phase shifts
        None => |_name: &str, _angle: f64| -> f64 { 0. },
    };

    // Gate sets to be cancelled
    /* Traverse each qubit to generate the cancel dictionaries
     Cancel dictionaries:
      - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        the value is the list of gates that share the same gate type, qubit, commutation set.
      - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        sec_commutation_set_id), the value is the list gates that share the same gate type,
        qubits and commutation sets.
    */
    let (commutation_set, node_indices) =
        analyze_commutations(dag, commutation_checker, approximation_degree)?;
    let mut cancellation_sets = IndexMap::with_hasher(::foldhash::fast::RandomState::default());

    (0..dag.num_qubits() as u32).for_each(|qubit| {
        let wire = Qubit(qubit);
        let wire_commutation_set = &commutation_set[qubit as usize];
        if !wire_commutation_set.is_empty() {
            for (com_set_idx, com_set) in wire_commutation_set.iter().enumerate() {
                if let Some(&nd) = com_set.first() {
                    if !matches!(dag[nd], NodeType::Operation(_)) {
                        continue;
                    }
                } else {
                    continue;
                }
                for node in com_set.iter() {
                    let instr = match &dag[*node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set."),
                    };
                    let num_qargs = dag.get_qargs(instr.qubits).len();
                    // no support for cancellation of parameterized gates
                    if instr.is_parameterized() {
                        continue;
                    }
                    if let Some(op_gate) = instr.op.try_standard_gate() {
                        if num_qargs == 1 && SUPPORTED_GATES.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }

                        if num_qargs == 1 && Z_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::ZRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        if num_qargs == 1 && X_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::XRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        // Don't deal with Y rotation, because Y rotation doesn't commute with
                        // CNOT, so it should be dealt with by optimized1qgate pass
                        if num_qargs == 2 && dag.get_qargs(instr.qubits)[0] == wire {
                            let second_qarg = dag.get_qargs(instr.qubits)[1];
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire, second_qarg],
                                    com_set_index: com_set_idx,
                                    second_index: node_indices[second_qarg.index()]
                                        .get(node)
                                        .copied(),
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                    }
                }
            }
        }
    });

    for (cancel_key, cancel_set) in &cancellation_sets {
        if cancel_set.len() > 1 {
            if let GateOrRotation::Gate(g) = cancel_key.gate {
                if SUPPORTED_GATES.contains(&g) {
                    for &c_node in &cancel_set[0..(cancel_set.len() / 2) * 2] {
                        dag.remove_op_node(c_node);
                    }
                }
                continue;
            }
            if matches!(cancel_key.gate, GateOrRotation::ZRotation) && z_var_gate.is_none() {
                continue;
            }
            if matches!(
                cancel_key.gate,
                GateOrRotation::ZRotation | GateOrRotation::XRotation
            ) {
                let mut total_angle: f64 = 0.0;
                let mut total_phase: f64 = 0.0;
                for current_node in cancel_set {
                    let node_op = match &dag[*current_node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set run."),
                    };
                    let node_op_name = node_op.op.name();

                    let (node_angle, phase_shift) = if ROTATION_GATES.contains(&node_op_name) {
                        let node_angle = match node_op.params_view().first() {
                            Some(Param::Float(f)) => *f,
                            _ => {
                                return Err(CommutationCancelError::RotationalParameter(
                                    node_op.op.clone(),
                                ));
                            }
                        };
                        let phase_shift = z_phase_shift(node_op_name, node_angle);
                        Ok((node_angle, phase_shift))
                    } else {
                        match node_op_name {
                            "t" => Ok((FRAC_PI_4, z_phase_shift("p", FRAC_PI_4))),
                            "tdg" => Ok((-FRAC_PI_4, -z_phase_shift("p", FRAC_PI_4))),
                            "s" => Ok((FRAC_PI_2, z_phase_shift("p", FRAC_PI_2))),
                            "sdg" => Ok((-FRAC_PI_2, -z_phase_shift("p", FRAC_PI_2))),
                            "z" => Ok((PI, z_phase_shift("p", PI))),
                            "x" => Ok((PI, FRAC_PI_2)),
                            "sx" => Ok((FRAC_PI_2, FRAC_PI_4)),
                            "sxdg" => Ok((-FRAC_PI_2, -FRAC_PI_4)),
                            _ => Err(CommutationCancelError::OperationUndefinedAngle(
                                node_op_name.to_owned(),
                            )),
                        }
                    }?;
                    total_angle += node_angle;
                    total_phase += phase_shift;
                }

                if is_multiple_of_pi(total_angle, 4.) {
                    // if the angle is close to a 4-pi multiple (from above or below), then the
                    // operator is equal to the identity
                } else if is_multiple_of_pi(total_angle, 2.) {
                    if cancel_key.gate == GateOrRotation::XRotation
                        || matches!(z_var_gate, Some(&StandardGate::RZ))
                    {
                        // a 2-pi multiple has a phase of pi: RX(2pi) = RZ(2pi) = -I = I exp(i pi)
                        total_phase -= PI;
                    }
                } else if cancel_key.gate == GateOrRotation::ZRotation {
                    let z_gate = z_var_gate.unwrap();
                    dag.insert_1q_on_incoming_qubit((*z_gate, &[total_angle]), cancel_set[0]);
                } else {
                    // cancel_gate.key is either ZRotation or XRotation in this block it is an
                    // XRotation.
                    if x_supported && is_multiple_of_pi(total_angle, 1.) {
                        let num_x = (total_angle / PI).round();
                        total_phase -= FRAC_PI_2 * num_x;
                        dag.insert_1q_on_incoming_qubit((StandardGate::X, &[]), cancel_set[0]);
                    } else if sx_supported && is_multiple_of_pi(total_angle, 0.5) {
                        let num_sx = (total_angle / FRAC_PI_2).round();
                        total_phase -= FRAC_PI_4 * num_sx;
                        for _ in 0..(num_sx as i64).rem_euclid(4) {
                            dag.insert_1q_on_incoming_qubit((StandardGate::SX, &[]), cancel_set[0]);
                        }
                    } else {
                        dag.insert_1q_on_incoming_qubit(
                            (StandardGate::RX, &[total_angle]),
                            cancel_set[0],
                        );
                    }
                }

                dag.add_global_phase(&Param::Float(total_phase))?;

                for node in cancel_set {
                    dag.remove_op_node(*node);
                }
            }
        }
    }

    Ok(())
}

/// Checks if angle is an integer multiple of ``factor * PI``.
fn is_multiple_of_pi(angle: f64, factor: f64) -> bool {
    let modulo = angle / (factor * PI);
    let remainder = modulo.rem_euclid(1.0);
    abs_diff_eq!(remainder, 0., epsilon = _CUTOFF_PRECISION)
        || abs_diff_eq!(remainder, 1., epsilon = _CUTOFF_PRECISION)
}

pub fn commutation_cancellation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_cancel_commutations))?;
    Ok(())
}
