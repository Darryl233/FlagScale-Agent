<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Graph Capture and Compilation

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about graph capture, compilation boundaries, or recompilation cost. A cheap configuration comparison does not require a full profile first.
Reuse the confirmed target call and NPU support scope. For mechanisms, read `know-ascend-training`: `ascend_training/graph-execution.md` as needed. For data supply or runtime task queues, use the [data and host method](data-and-host.md); for fusion or backend selection, use the [operator method](operator.md).

## Generate candidates

Start from the parent recipe and actual call path. Choose a supported path and specify the target module, input conditions, and boundary to change:

| Direction | Concrete operation |
| --- | --- |
| Local training graph capture | Once support in the current TorchNPU version is confirmed, insert `graphed = torch_npu.npu.make_graphed_callables(module, sample_args)` at the target static module call and replace that call with the returned callable. Match sample tensors to actual shape, dtype, and `requires_grad`; retain original parameter objects and the surrounding training loop. If a graph path already exists, compare capture scope or shape sets rather than wrapping it again. |
| Supported training compilation | Once the current NPU backend is known to support the target forward and backward paths, use `torch.compile(target, backend=npu_backend, ...)` at the target callable. `npu_backend` means a confirmed backend object/name; do not assume the default backend is supported. Adjust the compilation boundary or `dynamic` based on existing graph-break/recompilation evidence, and retain an eager comparison. Do not treat inference-oriented `npugraph_ex` as a general training entrypoint. |

These are starting points, not limits on scope or order. With evidence and budget, investigate graph breaks, repeated preparation, and overhead between graphs to find a better compilation or capture boundary.
Preserve layout, MBS/GBS, precision, model, and data semantics. For a code candidate, name the target function and change; implement it when selected by the main loop and source edits are authorized. Reuse confirmed support and investigate only gaps relevant to the current candidate.

## Additional checks

- **Graph/compilation:** Separate warmup, compilation/capture, and replay. Confirm actual execution, graph breaks/fallback, input updates, and graph resources. For a new boundary, cover varying inputs, multiple updates and associated gradient accumulation, RNG/dropout, and recomputation. Gradients, RNG changes, or state side effects from preparation must not contaminate the comparison's initial state. Report one-time preparation cost separately; count repeated compilation/capture as real cost.

Explain preparation cost and steady-state gains separately. Attribute gains from a valid local fallback only to the path that actually executes. If fallback or repeated capture persists, use its cost to decide whether to narrow or fix the boundary. Successful capture does not prove faster training.
