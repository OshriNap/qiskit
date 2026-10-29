# Commuting gate reordering and five CX cancellation

This review branch combines two local Rust optimizations in existing Qiskit passes. The implementation was built and tested on CPU on 28 October 2026. It reduces measured native two-qubit depth on several existing benchmark families, with compilation overhead on some inputs.

**The original full `confirm-profile` verdict is `CONSTRAINT_VIOLATION`, not PASS.** The descriptive weighted depth reduction is **1.101%**, but the original audit prerequisites did not pass and the official objective is unavailable. The harness skipped its full correctness and cost stages. This branch is a prototype for review, with the failures and measurement limits retained below.

## Implementation

The code commit changes nine files against Qiskit `73aa14d2b8b423335a72cb1b8f940249029cb7df`. The following commit adds only this review material. The [provenance](provenance.json) identifies the code commit and the SHA-256 of every measured and published source file. The older `codex/comparator-prefix` branch is a separate contribution and is not included here.

1. **Five CX gates become three.** For distinct wires `a`, `b`, `c`, let `A = CX(a,b)`, `B = CX(b,a)`, and `D = CX(c,a)`. In circuit execution order, `A B A D B` equals `B A D`. The matcher also handles reverse order and simultaneous reversal of all CX directions. It deletes the two endpoints, preserving the middle gates. A bounded five-operation window must be consecutive on both repeated wires; overlapping windows are reserved before mutation. Control-flow and unsupported variable contexts are skipped. This is an exact identity with a local saving of two CX gates; later whole-pipeline choices can still change the final gate count.

2. **Reorder adjacent commuting interactions.** A single forward DAG scan can interchange adjacent standard CX/CZ operations sharing one wire when they commute exactly and the arrival-depth condition advances the second interaction without delaying the participating wires. It makes at most one interchange per visited operation. This changes dependency order without directly removing a gate. It does not introduce an internal fixed-point loop.

`CommutativeCancellation` performs ordinary cancellation, then five-CX endpoint deletion, then depth reordering using the resulting graph. Its existing Python wrapper enables reordering when a target is supplied and five-CX cancellation when that target includes CX. The private native flags default to false. The same five-CX matcher prepares CX-target circuits before analysis/synthesis in `TwoQubitPeepholeOptimization`, exposing useful two-qubit runs earlier in the existing pipeline.

| Changed file | Purpose |
| --- | --- |
| [dag_circuit.rs](../../crates/circuit/src/dag_circuit.rs) | Extract the existing Python-facing `swap_nodes` implementation into a Rust method; the Python wrapper delegates to it. |
| [commutation_cancellation.rs](../../crates/transpiler/src/passes/commutation_cancellation.rs) | Bounded five-CX matcher, endpoint deletion, one-scan depth reordering and native entry orchestration. |
| [two_qubit_peephole.rs](../../crates/transpiler/src/passes/two_qubit_peephole.rs) | Reuse the matcher before existing analysis/synthesis; preserve the Rust/Python entry contracts. |
| [commutative_cancellation.py](../../qiskit/transpiler/passes/optimization/commutative_cancellation.py) | Target-dependent enablement and documentation. |
| [two_qubit_peephole.py](../../qiskit/transpiler/passes/optimization/two_qubit_peephole.py) | Document the bounded preparation. |
| [test_commutative_cancellation.py](../../test/python/transpiler/test_commutative_cancellation.py) | Reorder, identity, boundary and composition regressions, including operator equivalence. |
| [test_two_qubit_peephole.py](../../test/python/transpiler/test_two_qubit_peephole.py) | Five-CX preparation regressions. |
| Two release notes in [releasenotes/notes](../../releasenotes/notes) | Separate notes for commuting-depth reordering and five-CX cancellation. |

There are no changes to preset pass managers, pipeline ordering, the convergence policy, `MinimumPoint`, public constructor options, the expression engine, dependencies or benchmark inputs. The five-CX identity has prior art in optional templates; the contribution here is its bounded integration into these existing passes. This publication does not establish novelty or absence of concurrent work.

## Metrics and benchmark configuration

The harness is [qiskit-transpile-bench at 2b617646](https://github.com/ophirazulai/qiskit-transpile-bench/tree/2b617646546cccdc8066d0f08c2a13e849ef6381). Its `D2` metric is dependency depth counting native two-qubit operations. Other operations synchronize the participating wires with zero depth weight. `N2` counts native two-qubit operations. Ordinary circuit depth also counts single-qubit layers and is a different metric. No quantum hardware runtime or fidelity was measured.

The full original `confirm-profile` used 209 quality cases, including 133 scored cases across eight families and optimization levels 0–3. Every scored case used the original 100 seeds. In total, 31,200 quality observations form **15,600 baseline/candidate pairs**. Both native extensions were independently rebuilt with the harness's original thin-LTO release configuration, Python 3.11.16 and Rust 1.91. The original weights, inputs, guards, thresholds and stage gates were retained.

## Full confirm results

The following are **descriptive calculations using the original estimator on the saved measurements**, not an accepted objective or replacement verdict. Each restricted panel is weighted separately; the full score is not an arithmetic average of the level rows.

| Scored panel | D2 ratio candidate / baseline | D2 reduction | N2 reduction |
| --- | ---: | ---: | ---: |
| All 133 scored cases | 0.988988 | 1.101% | 0.426% |
| Level 0 | 1.000000 | 0% | 0% |
| Level 1 | 1.000000 | 0% | 0% |
| Level 2 | 0.977621 | 2.238% | 0.778% |
| Level 3 | 0.976602 | 2.340% | 1.016% |

The full-panel upper two-standard-error ratio is **0.989578**, below the original numerical threshold of 0.99. Five of eight families have a negative upper-two-SE effect: QAOA, arithmetic/MCX, Bernstein–Vazirani, ring/SU2 and QUEKO. All leave-one-family-out D2 ratios are below one. These numerical observations do not establish that the unrun correctness/cost stages would pass.

| Metric over all 15,600 pairs | Improved | Tied | Regressed |
| --- | ---: | ---: | ---: |
| D2 | 3,208 | 12,392 | 0 |
| N2 | 1,912 | 13,687 | 1 |

The sole N2 regression is **QUEKO BSS53, level 3, seed 92: 1,005 → 1,031 gates**, while D2 improves **189 → 180**. Zero observed D2 regressions on this panel is not a universal guarantee.

Previously studied inputs reproduced in this full run, using arithmetic means over 100 seeds per row:

| Input | Level | Mean D2 before → after | D2 reduction | Mean N2 before → after |
| --- | ---: | ---: | ---: | ---: |
| BV50 / Rochester53 | 2 | 192.02 → 172.97 | 9.92% | 196.86 → 181.92 |
| BV50 / Rochester53 | 3 | 193.78 → 174.54 | 9.93% | 199.74 → 185.53 |
| QUEKO BSS53 + SABRE / Rochester53 | 2 | 230.24 → 216.47 | 5.98% | 1104.81 → 1073.01 |
| QUEKO BSS53 + SABRE / Rochester53 | 3 | 197.72 → 180.58 | 8.67% | 947.01 → 900.83 |
| QUEKO BIGD20 + SABRE / Tokyo20 | 2 | 20.39 → 18.91 | 7.26% | 21.72 → 21.72 |
| QUEKO BIGD20 + SABRE / Tokyo20 | 3 | 19.77 → 18.36 | 7.13% | 21.25 → 21.19 |
| Ripple adder10 / grid5×5 | 2 | 206.02 → 204.31 | 0.83% | 349.95 → 349.93 |
| Ripple adder10 / grid5×5 | 3 | 201.73 → 200.11 | 0.80% | 347.77 → 347.74 |

All three scored `iterations-profile` inputs—QFT100, square-Heisenberg100 and QAOA100 at level 2 on the CZ target—were matched by exact input/target hashes and options within this confirm run. They have **100 D2 ties and 100 N2 ties each**. This is not a separate iterations-profile run. There was no direct Benchpress run of this combined candidate, and the confirm workload was used during development rather than held out.

## Official validation outcome

The original quality gate remained closed and `decide` returned exit 20, `CONSTRAINT_VIOLATION`. The official objective is empty. Full correctness and cost were **skipped**, and optional upstream unit tests were not started in this harness session.

| Check | Baseline | Candidate |
| --- | ---: | ---: |
| C0 structural checks verified | 15,600 | 15,600 |
| C1-lite verified | 619 | 621 |
| C1-lite unverified at the 300-second verifier timeout | 11 | 9 |
| C6 layout mismatches | 1,041 | 1,012 |
| Determinism audit differences, out of 780 per arm | 74 | 66 |

Most C6 differences preserve the logical input mapping and physical routing but change appended-ancilla index order. The remaining 29 baseline / 28 candidate cases involve the placement of an isolated input in QAOA32. Of 140 audit differences, 136 change only the layout: 135 preserve logical maps/routing, and one baseline BIGD20 case exchanges unused inputs. Four QAOA32 audit outputs change their canonical output hash as well as layout; the original structural grader finds the same D2/N2. These diagnostics do not replace full semantic equivalence certification. The original statuses are retained.

A local shell wrapper emitted a parse error after the quality stage had committed exit 0 / complete. Its output-printing code had been edited while the shell was still running, consistent with a stale read offset. Fresh subsequent stage invocations completed, with correctness/cost skipped and decide returning 20. The wrapper error was recorded separately; the original quality results were not rerun or rewritten. Windows Git pointers also required source-only Linux exports before the clean run; every exported byte and executable mode was checked against the frozen source snapshots.

## Earlier four arm comparison and compile time

A separate native study compared baseline, reorder only, five-CX only and combined code on 1,204 unique case/level/seed rows. The combined result had 819 D2 wins and no D2 losses against baseline, but three D2 losses against the better standalone result. There were also 67 total-gate and 12 ordinary-depth regressions against baseline. Local improvements do not guarantee dominance after the remaining passes.

The combined BV case uses three optimization rounds at level 2 and three/five rounds at level 3 (61/39 of 100 seeds), versus two for the latest five-CX-only candidate. Reordering exposes further cancellations; the unchanged convergence and `MinimumPoint` policies may then wait and restore an earlier ordinary-depth/size optimum. This also explains the BSS53 seed-92 N2 regression.

The earlier timing study used **372 balanced interleaved quartets / 1,488 measured native runs** on a shared CPU host. It timed `pm.run(qc)`, excluding pass-manager construction, callbacks, metrics and loading. Ratios are medians of paired process-CPU ratios. These binaries/configuration are separate from the full confirm builds, whose cost stage did not run.

| Case | Level | Combined / baseline CPU time |
| --- | ---: | ---: |
| BV50 | 2 | 0.9996× |
| BV50 | 3 | 1.1596×, about 16% slower |
| BSS53 | 2 | 0.9848× |
| BSS53 | 3 | 1.0176× |
| BIGD20 | 2 | 0.9592× |
| BIGD20 | 3 | 1.0742× |
| Ripple adder10 | 2 | 0.9949× |
| Ripple adder10 | 3 | 0.9913× |

Three routing seeds (0, 7, 42), two warmups and 12 measured repeats were used per group; BV level 3 also included seed 2 to exercise five rounds. That selected stress seed was 28.00% slower than baseline, versus 13.72% for the selected three-round stratum. The BV level-3 ratio's interquartile range was 0.987–1.345×. Shared-host variation limits small timing claims; these selected seeds do not estimate the entire 100-seed timing population. There is no accepted full-confirm runtime or memory result.

## Repository and semantic checks

The earlier source-identical combined build passed 984 Python tests, 33 threaded subset reruns, 47 Rust transpiler tests, 83 Rust core tests excluding one known failure, and an isolated native-entry test. The deep expression test `parameter::symbol_expr::test::test_eq_exact_binary_deep` failed on both the candidate and the previously built, hash-verified baseline test executable. It was explicitly excluded from the remaining core run. Black, Ruff, cargo fmt and diff whitespace checks passed.

Additional CPU validation covered 8,100 native/reference comparisons, 7,776 exhaustive five-CX binary-map cases, 324 local operator comparisons and 60 pipeline operator comparisons. Maximum numerical errors were approximately 5.80e-16 and 2.19e-15 for the two operator sets. These targeted checks do not certify every large benchmark output. Of 11 examined changed-final-mapping rows, one tractable QAOA8 case was simulated for all four arms and three random states each; the other ten were explicitly not simulated.

Implementation and experiments were developed with coding agents; the source, measurements and limitations are retained for human review. No repository tests or benchmarks were rerun just to publish this branch. Eight files, including every implementation file, match the measured bytes exactly. Only `test_two_qubit_peephole.py` was normalized from CRLF to LF for publication, matching the existing reviewed Git diff; its source text is unchanged. Both hashes and the normalization are recorded in provenance.

## Reproducing and inspecting the evidence

Run `python experiments/commuting-depth-cx5/verify_evidence.py` from this branch to check the public file hashes, paired counts, per-case means and published code hashes. This checks evidence consistency, not circuit correctness.

For a new benchmark run, use a Linux checkout of the **code commit** in `provenance.json` for the candidate and baseline commit `73aa14d2b8b423335a72cb1b8f940249029cb7df` for the baseline. The code commit excludes these review documents, preserving the original nine-path source scope. Check out harness `2b617646546cccdc8066d0f08c2a13e849ef6381`, install its documented prerequisites, and run its original workflow:

```bash
uv sync --all-extras
mkdir -p /path/to/new-store
uv run qiskit-transpile-bench compile \
  --baseline /path/to/qiskit-baseline \
  --evolved /path/to/qiskit-candidate \
  --store /path/to/new-store --profile confirm-profile \
  --results-root /path/to/new-session
uv run qiskit-transpile-bench quality --results-root /path/to/new-session
uv run qiskit-transpile-bench correctness --results-root /path/to/new-session
uv run qiskit-transpile-bench cost --results-root /path/to/new-session
uv run qiskit-transpile-bench decide --results-root /path/to/new-session
```

Choose a session path that does not exist. Gates can skip later stages. The original CLI's final cleanup removes bulk artifacts; archive needed evidence first. In the measured run, the unchanged official `decide()` function was called directly to retain those artifacts, with no change to decision rules.

The public files include all 15,600 metric pairs, case means, descriptive panels, the original decision, original manifest/policy, earlier four-arm summary/timing observations, validation counts and code/build hashes. Gzip files are ordinary compressed JSON/CSV. In copied JSON evidence, only local filesystem prefixes are replaced by descriptive placeholders; provenance records the original and public hashes. Full output circuits, environments and local build/verifier logs remain in the local evidence archive and are not part of this branch. The public selection cannot independently re-prove every verifier result.

Outstanding acceptance work includes resolving the baseline/candidate validation discrepancies, obtaining a valid full cost result and assessing the extra scans, graph-method extraction and bounded three-qubit preparation inside the two-qubit peephole pass. Publishing this review branch does not resolve those points.
