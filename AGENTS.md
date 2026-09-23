# AGENTS.md

## Project overview

Loopy is a Python library for transformation-based generation of high-performance array-oriented code for CPUs and GPUs. Users describe a computation and its iteration domains, then apply explicit transformations for tiling, parallelization, prefetching, layout changes, unrolling, vectorization, and instruction-level parallelism.

Loopy is intentionally not a general-purpose programming language. Its main application areas include linear algebra, convolutions, N-body interactions, finite-element/finite-difference solvers, and other structured array computations.

Important dependencies and representations:

- `namedisl` and `islpy`: Presburger/polyhedral sets, maps, (possibly piece-wise) quasi-affine expressions.
- `pymbolic`: scalar and array expression trees and mapper infrastructure.
- `pytools`: caching, persistent hashes, and utility infrastructure.
- `cgen`/`genpy`: generated C-family and Python syntax trees.
- `pyopencl`, optionally: OpenCL execution and runtime integration.

The package requires Python 3.10 or newer. Static-analysis configuration targets Python 3.12.

## Repository layout

### Core package

- `loopy/__init__.py`
  - Main public API and re-exports.
  - Check whether a new public class or transform must be exported here.

- `loopy/kernel/__init__.py`
  - `LoopKernel`, kernel state, core kernel-level queries, grid sizing, variable lookup.

- `loopy/kernel/data.py`
  - Kernel arguments, temporaries, address spaces, iname tags, hardware-axis tags, and related descriptors.
  - Current `AddressSpace` values are `PRIVATE`, `LOCAL`, and `GLOBAL`.

- `loopy/kernel/array.py`
  - Current array shape/layout representation.
  - `ArrayBase`, array dimension implementation tags, stride conversion, and access lowering through `get_access_info`.
  - This is a central file for array changes, but many consumers live elsewhere.

- `loopy/kernel/instruction.py`
  - Instruction classes, assignment/call/barrier semantics, dependency and access queries.

- `loopy/kernel/creation.py`
  - Kernel construction/parsing, temporary creation, shape inference, slice realization, and creation-time normalization.

- `loopy/kernel/tools.py`
  - Kernel analyses and utilities, array lookup/change helpers, shape guessing, alias classes, and graph-like queries.

- `loopy/kernel/function_interface.py`
  - Callable descriptors and specialization.
  - `ArrayArgDescriptor`, `InKernelCallable`, `CallableKernel`, subarray argument descriptor inference.

- `loopy/translation_unit.py`
  - Translation-unit representation, callable table, callable resolution, and specialization context.

### Expressions and polyhedral analysis

- `loopy/expression.py`
  - Expression-related checks and transformations.

- `loopy/symbolic.py`
  - Pymbolic mappers, affine conversion, access maps/ranges, `SubArrayRef`, dependency extraction, and overlap analysis.
  - Many analyses construct maps from instruction domains to array index tuples here.

- `loopy/isl_helpers.py`
  - Utilities around ISL/namedisl bounds, extrema, projections, and set/map manipulation.

- `loopy/types.py`, `loopy/typing.py`, `loopy/type_inference.py`
  - Loopy types, shared type aliases, and type inference.

### Transformations

- `loopy/transform/`
  - User-facing and internal kernel transformations.
  - Important modules include:
    - `data.py`: array-axis tags, temporary address-space setters, data-layout-related entry points.
    - `iname.py`: iname splitting/tagging and loop transformations.
    - `precompute.py`: prefetch/precompute transformations.
    - `privatize.py`: temporary duplication/private axes.
    - `realize_reduction.py`: reduction lowering.
    - `callable.py`: callable-kernel transformations.
    - `padding.py`, `buffer.py`, `array_buffer_map.py`: storage/layout changes.
    - `save.py`: save/reload across scheduling boundaries.
  - Many transform functions use `@for_each_kernel` so they operate on each callable kernel in a translation unit.

### Preprocessing, checks, scheduling, and code generation

- `loopy/preprocess.py`
  - Pre-codegen normalization.
  - Handles separate arrays, offsets/strides, ILP realization, temporary address-space inference, target preprocessing, and callable argument descriptor inference.
  - Ordering comments in this file are significant; do not reorder passes casually.

- `loopy/check.py`
  - Structural, bounds, access-ordering, race, callable, and pre-schedule checks.
  - Bounds checking and several address-space assumptions currently depend on tuple shapes and old address spaces.

- `loopy/schedule/`
  - Schedule generation and schedule-time dependency/barrier reasoning.
  - `schedule/tools.py` contains access-race logic and generated-subkernel argument decisions.
  - `schedule/__init__.py` contains dependency tracking and schedule generation machinery.

- `loopy/codegen/`
  - Target-independent code-generation state and control/loop/instruction lowering.

- `loopy/target/`
  - Target-specific AST generation and execution wrappers.
  - `c/`: common C-family target and expression lowering.
  - `opencl.py`: OpenCL declarations, atomics, address qualifiers.
  - `cuda.py`: CUDA declarations and shared/global/private lowering.
  - `ispc.py`: ISPC target, including private-lane handling.
  - `python.py`: Python target.
  - `pyopencl.py`, `pyopencl_execution.py`: OpenCL code generation and host execution.
  - `execution.py`: common runtime argument validation, shape/stride checks, and parameter inference.

### Frontends and library callables

- `loopy/frontend/fortran/`
  - Fortran frontend.

- `loopy/library/`
  - Built-in callable/reduction/random functionality.

### Documentation and tests

- `doc/`
  - Sphinx/reStructuredText reference and tutorial documentation.
  - `ref_kernel.rst`, `ref_call.rst`, and `ref_internals.rst` are common destinations for core changes.

- `test/`
  - Main pytest suite.
  - `test_loopy.py` is broad and historical.
  - `test_callables.py`, `test_transform.py`, `test_target.py`, `test_c_execution.py`, and `test_linalg.py` are useful focused suites.

- `examples/`
  - User-facing and extension examples. Check these when modifying public callable or layout APIs.

- `proto-tests/`
  - Prototypes excluded from the default pytest run.

## High-level processing pipeline

The exact order varies by API, target, and kernel state, but the main flow is:

1. Create a `LoopKernel`/`TranslationUnit` from domains and instructions.
2. Infer or normalize arguments, temporaries, shapes, dependencies, and types.
3. Apply user-requested transformations.
4. Resolve callable names and specialize callable kernels.
5. Preprocess reductions, substitutions, array metadata, ILP/private storage, address spaces, and target-specific details.
6. Run structural, bounds, race, and scheduling checks.
7. Generate/linearize a schedule, including barriers and generated device subkernels where required.
8. Lower expressions and instructions into target ASTs.
9. Generate device code and, for execution targets, host-side argument validation/allocation wrappers.

Do not assume a `LoopKernel` is in the same state at all entry points. Check `KernelState` and nearby invariants before adding logic.

## IR and coding conventions

### Immutability

Kernel IR is treated as immutable:

- use `.copy(...)` rather than mutation;
- use frozen dataclasses patterns consistent with neighboring code (when possible, replace instances of the old `ImmutableRecord` with dataclasses);
- use `constantdict` where callable/descriptor maps must be hashable;
- ensure constructor normalization remains compatible with `.copy(...)`, which may reconstruct through constructors.

When adding semantic fields, update:

- equality;
- hashing and persistent hashing;
- expression mapping/substitution;
- dependency collection;
- stringification/repr;
- pickle/reproducer behavior where relevant.

### Expressions

Scalar expressions are usually Pymbolic expressions, not Python ASTs. Use existing mapper infrastructure and symbolic helpers rather than manually recursing through expression trees.

Be explicit about whether an expression is:

- compile-time constant;
- quasi-affine/polyhedrally representable;
- dependent only on integral value arguments;
- valid in caller or callee parameter namespaces.

### Named ISL spaces

Loopy uses named dimensions extensively. When comparing or composing sets/maps:

- align by name rather than relying on positional parameter order;
- preserve logical dimension names;
- account for kernel assumptions and instruction domains;
- distinguish parameters, input dimensions, and output dimensions;
- avoid silently projecting out information needed for correctness.

### Public APIs

Public functions/classes are commonly re-exported from `loopy/__init__.py`. Changes to constructors, descriptors, or transform syntax may affect third-party code even if symbols are not re-exported.

Prefer compatibility conversion at API boundaries over maintaining two canonical internal representations.

### Warnings and errors

Use existing `LoopyError` subclasses and `warn_with_kernel` patterns. Diagnostics should name the array/instruction/callable and state which invariant could not be established.

Conservative analysis should fail or report “may overlap/may race” rather than silently accepting an unsafe program.

## Callable-kernel considerations

Callable kernels are specialized from argument descriptors. Current array descriptors carry tuple shape, address space, and dim tags. `SubArrayRef` is the required representation for passing array regions.

When changing callable array semantics:

- update `ArrayArgDescriptor.map_expr` and `.depends_on`;
- update `get_arg_descriptor_for_expression`;
- update `CallableKernel.with_descrs`;
- check nested callable specialization and translation-unit callable deduplication;
- preserve caller/callee parameter namespace translation;
- inspect third-party extension examples such as custom `InKernelCallable.with_descrs` implementations;
- ensure unresolved alternatives or compatibility representations do not reach code emission.

Callable specialization identity is hash-sensitive. New descriptor values must be immutable and deterministic.

## Race and alias analysis cautions

There are multiple race-related mechanisms:

- per-instruction write-race warnings in `loopy/check.py`;
- variable access ordering checks in `loopy/check.py`;
- access range overlap in `loopy/symbolic.py`;
- schedule-time `WriteRaceChecker` and barrier reasoning in `loopy/schedule/tools.py`;
- automatic temporary address-space inference in `loopy/preprocess.py`.

The current per-instruction check often uses syntactic occurrence of an iname in a subscript. Occurrence is not injectivity: `i % 2` and `i-i` can collide. Avoid extending this shortcut. Prefer two-copy ISL collision queries when working in this area.

Current access overlap is primarily expressed in logical source-level indices. Noninjective physical layouts can invalidate logical-disjointness reasoning. The planned layout model excludes noninjective layouts and makes storage-instance axes explicit.

Base-storage aliases require separate care. Distinct temporary names may share physical storage even when their logical index spaces differ. Preserve conservative behavior unless offsets/layouts are composed into a common physical coordinate.

## Generated subkernels and storage lifetime

Global barriers can split one logical kernel into multiple generated device programs. Current code uses `AddressSpace.GLOBAL` to decide which temporaries are host/device allocated and passed between generated subkernels.

Any storage-model change must preserve these distinctions:

- persistent/global storage may cross a launch boundary;
- local/shared storage may not;
- private/work-item storage may not;
- sequential iname-private storage may not escape its lifetime;
- initialized global constants have separate declaration behavior.

Audit:

- `loopy/schedule/tools.py:_should_temp_var_be_passed`;
- generated-subkernel argument collection;
- checks for definitions across subkernels;
- PyOpenCL global temporary allocation/release;
- base-storage allocation;
- local-memory accounting.

## Common change maps

### Adding or changing an IR field

Check:

- constructor normalization;
- type annotations;
- `.copy(...)` reconstruction;
- equality and hash;
- persistent hash;
- expression mapping;
- dependency/supporting-name collection;
- stringification;
- callable descriptors;
- pickle/Python reproducer;
- tests.

### Changing array access semantics

Check:

- `get_access_info` or its replacement;
- C-family expression lowering;
- image lowering;
- vector and separate behavior;
- `indexof` support;
- bounds checking;
- race/overlap maps;
- callable `SubArrayRef` composition;
- atomics and volatile casts.

### Changing address-space/storage semantics

Check:

- temporary scope inference;
- `LoopKernel.global_var_names`/`local_var_names`;
- race checks and barrier scopes;
- target declarations;
- generated-subkernel lifetime;
- host allocation;
- local-memory statistics;
- save/reload transformations.

### Changing runtime argument semantics

Check:

- `loopy/target/execution.py`;
- C execution wrappers;
- PyOpenCL execution wrappers;
- input validation;
- output allocation;
- shape/stride parameter inference;
- empty and singleton dimensions;
- `skip_arg_checks`.

## Validation workflow

Start with the narrowest relevant test, then broaden.

Typical commands from the repository root:

```sh
python -m pytest test/test_callables.py -q
python -m pytest test/test_loopy.py -q
python -m pytest test/test_target.py -q
python -m pytest -q
```

For one test:

```sh
python -m pytest test/test_callables.py::test_name -q
```

Static checks configured by the repository include:

```sh
ruff check loopy test
basedpyright
```

The repository also contains strict `pyrefly` configuration, though availability depends on the development environment.

Execution tests for OpenCL/CUDA/ISPC may require optional packages, compilers, drivers, and hardware. If those are unavailable, run IR/code-generation tests and state which runtime validation was not performed.

Do not claim a command passed unless it was run successfully. If the suite has unrelated failures, report them separately and avoid changing unrelated behavior merely to obtain a clean run.

## Implementation style

- Whenever possible, prefer constructor functions (`make_xyz`) to having any logic at all (let alone complex logic) in an object constructor. The ideal constructor is the trivial/dataclass-generated one. Backward compatibility constrains this somewhat, but use this goal as design guidance.

## Test-writing guidance

- Prefer a focused new test module for a major subsystem instead of continually expanding `test_loopy.py`.
- Inspect normalized IR directly where that proves semantics more robustly than matching generated text.
- Use generated-code assertions for target syntax and qualifiers.
- Use execution tests for representative end-to-end storage and wrapper behavior.
- Include symbolic parameters, empty extents, singleton extents, and nonzero bases where relevant.
- For polyhedral behavior, test both a valid point and a point inside a bounding box but outside the actual set.
- For races, include injective and colliding maps, not only presence/absence of an iname in the index.
- For compatibility changes, retain tests using the old public spelling as well as tests for the new canonical API.

## Documentation expectations

Core behavior changes generally require updates to:

- `doc/ref_kernel.rst` for kernel/array/storage semantics;
- `doc/ref_call.rst` for callables and descriptors;
- `doc/ref_internals.rst` for preprocessing/code-generation invariants;
- `doc/ref_transform.rst` for public transformations;
- examples/tutorials for user-visible API changes.

Keep documentation terminology aligned with code. Distinguish logical shape, physical layout, storage instance, address space, and lifetime.

## Practical cautions

- Preprocessing and scheduling pass order has correctness dependencies; preserve and extend the existing ordering comments.
- Target-independent code should not encode OpenCL/CUDA syntax or ABI assumptions when a target hook is appropriate.
- Images are not ordinary global pointers and require opaque target-specific access/declaration behavior.
- Vector length three may have physical size four on OpenCL-like ABIs.
- `SeparateArrayArrayDimTag` currently affects argument ABI by materializing multiple arrays.
- `AddressSpace` is currently an ordered `IntEnum`; code uses `max()` as a scope join. Adding a non-lattice value without replacing those joins is unsafe.
- Rectangular runtime arrays and nonrectangular logical shapes are compatible only when the layout explicitly describes the physical container.
- A zero-dimensional array shape is a point/scalar, not an empty array.
- Avoid silent bounding-box conversion: it changes logical validity, not merely allocation.
