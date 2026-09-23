# Polyhedral Array Implementation Plan

## Scope

This plan implements the design in `ARRAY-DESIGN.md` in staged, reviewable changes. It covers Loopy core, public compatibility entry points, documentation, and correctness tests.

The plan is deliberately coarse. The design is expected to keep moving; phase boundaries, task lists, and file lists below are a sketch of sequencing, not a commitment. Go/no-go criteria are not yet meaningful and are not stated.

The final code-generation invariant is:

```text
every array holds a resolved Array = (named-set shape, layout),
and storage instances are explicit
```

## Guiding implementation rules

- Keep one canonical source of truth for shape and layout.
- Accept legacy forms at API boundaries and normalize immediately through dedicated constructor functions.
- Prefer `make_xyz` functions for conversion, normalization, validation, and other construction policy. Keep canonical records' constructors trivial and preferably dataclass-generated; backward-compatible class constructors and `copy()` shims should delegate to helpers rather than accumulating logic.
- Do not retain `tuple | namedisl.Set` as the resolved shape representation.
- Never derive a generic physical allocation size by ranging an address expression. Sizes are declared (`LinearLayout.size`) or structurally derived (`RectangularLayout`, `ImageLayout`).
- Derive `StorageKind` and `InstanceScope` from the layout; never store them beside it.
- Treat injectivity as a layout-provider contract for all layouts, not a responsibility of structural validation. Standard factories document why their constructions satisfy it.
- Keep race and dependency analysis at the logical level. Do not introduce physical-coordinate or footprint reasoning.
- No correctness decision may depend on hash-based identity of a shape or a layout.
- Preserve rectangular wrapper validation and parameter inference.
- Prefer conservative rejection over unsound race, alias, liveness, or storage-reuse reasoning.
- Add explicit pre-codegen invariants so partially migrated kernels fail early.

## Phase 0: Baseline and prototypes

### Objectives

- Record current behavior before changing representation.
- Prototype named-set hashing/copying and parameter substitution.
- Prototype the `Array`-as-a-held-value refactor on one class before committing.

### Tasks

1. Add characterization tests for:
   - tuple and string shape construction, including rejection of tuple entries equal to `None`;
   - scalar arrays with `shape=()`;
   - legacy top-level `shape=None` conversion to `auto` at supported inference entry points, rejection elsewhere, and direct `shape=auto`;
   - C/F/fixed-stride layouts;
   - vector and separate dim tags;
   - shape/stride runtime checks and parameter inference;
   - local/private/global/constant temporary and argument declarations;
   - image access;
   - subarray callable descriptors;
   - generated-subkernel temporary passing;
   - current race and barrier behavior.
2. Search external-facing examples and tests for direct tuple operations on `.shape`.
3. Prototype equality and persistent hashing for aligned `namedisl.Set`s. Equality is semantic; hashing uses `coalesce → detect_equalities → remove_redundancies` with parameters sorted by name. **Do not assume the normalization is canonical.** Add a fuzz test that generates semantically equal sets with different spellings and records how often the normalized forms differ; the result documents the expected rate of duplicated work, not a correctness bound.
4. Audit every dictionary, set, and cache keyed on something containing a shape or layout, and confirm that a hash disagreement between equal values causes only duplication or a cache miss. Base-storage grouping is the known case needing an explicit semantic comparison.
5. Prototype named-space alignment, persistent hashing, parameter substitution, pullback, totality/range checking, and singleton evaluation for `namedisl.PwAff` layout components.
6. Prototype the composition refactor on `TemporaryVariable` alone: hold an `Array`, add deprecated forwarding properties, and implement the `copy()` kwarg split. Measure the breakage before applying it to `ArrayArg`.

### Primary files

- `loopy/kernel/array.py`
- `loopy/kernel/data.py`
- `loopy/tools.py`
- `test/test_loopy.py`
- `test/test_callables.py`
- target runtime tests

## Phase 1: Canonical shape utilities

### Objectives

Introduce named-set shapes without yet removing legacy layout code.

### Tasks

1. Add the canonical shape type aliases:

   ```python
   ArrayShape = namedisl.Set | type[auto]
   ResolvedArrayShape = namedisl.Set
   ```

2. Add shape-construction and conversion helpers:
   - legacy tuple/string to named set, rejecting tuple entries equal to `None`;
   - top-level legacy `shape=None` to `auto` only for APIs that explicitly support exact inference, with targeted rejection everywhere else;
   - zero-dimensional point construction;
   - rank-known universe construction;
   - named-axis normalization;
   - shape parameter dependency extraction.
3. Add `get_default_shape_axis_name(i)` alongside `get_access_map_storage_names`. It returns `_lpy_s{i}` for zero-based axis `i`; `_lpy_` is protected, so it performs no collision avoidance. Make `get_access_map_storage_names` use this helper.
4. Add shape queries: `num_axes` (raising on `auto`), axis names, `rectangular_shape()`, scalar-shape predicate, empty-shape predicate.
5. Add stringification: NumPy shape notation only for a **zero-based** box; named-set notation for non-boxes and for boxes with a nonzero origin. `is_box` alone is not the predicate.
6. Route legacy shape inputs through the construction helper so `None` is never stored.
7. Replace `dim_names` canonical storage with set dimension names.
8. Define targeted errors for operations requiring a rectangular shape.

### Tests

- tuple-to-set conversion;
- deterministic `_lpy_s{i}` naming;
- rejection of `None` tuple entries;
- immediate top-level `shape=None` conversion or targeted rejection, with no stored `None`;
- zero-dimensional point/scalar semantics;
- empty sets;
- nonzero lower bounds;
- triangular, diamond, and union shapes;
- rectangular extraction success and failure;
- zero-based-box stringification versus named-set stringification for a nonzero-origin box;
- equality under parameter reordering;
- hash/equality divergence fuzz test from Phase 0 task 3;
- pickle and reproducer round trips.

## Phase 2: Polyhedral bounds and shape inference

### Objectives

Make shape-consuming analysis uniformly polyhedral.

### Tasks

1. Rewrite `_AccessCheckMapper.map_subscript` to build the access range, align parameters and output dimensions with the array shape, and check subset containment directly.
2. Update shape-parameter validation to inspect set parameters and require integral read-only `ValueArg`s for externally visible arrays.
3. Change `find_shapes_of_vars` to return access-range sets rather than base-index/extent tuples, and fail if any relevant access is unanalyzable instead of unioning only the successful ones.
4. Change `determine_shapes_of_temporaries` to retain exact access unions and diagnose an unremoved, access-free `auto` temporary.
5. Change `guess_var_shape` to return named sets.
6. Remove bounding-box *shape* inference and its options.
7. **Add bounding-box *layout* inference for `auto` temporaries**: shape is the exact access union; the default layout is `make_c_layout` over the bounding box of that union, with origins from the box's lower bounds. Without this, Phase 2 and Phase 5 are mutually unsatisfiable — an inferred temporary would have an exact shape and no way to be allocated. Document that this is the one legitimate appearance of a bounding box, and that it changes allocation only, never logical validity.
8. Migrate scalar-array checks from `shape == ()` to a helper.

### Primary files

- `loopy/check.py`
- `loopy/kernel/creation.py`
- `loopy/kernel/tools.py`
- `loopy/symbolic.py`
- `loopy/kernel/data.py`

### Tests

- in-bounds and out-of-bounds accesses for triangles and diamonds;
- a point inside the rectangular hull but outside the logical set;
- predicates narrowing an access into the valid set;
- symbolic shape parameters;
- union shapes;
- exact `auto` temporary inference;
- bounding-box layout over a triangular inferred shape: allocation covers the box, bounds checking still rejects the holes;
- access-free `auto` temporary removal or targeted failure;
- non-quasi-affine inference failure;
- scalar and empty arrays.

## Phase 3: Layout object framework

### Objectives

Introduce immutable layout values and access-lowering protocols while retaining adapters for current dim tags.

### Tasks

1. Define `LogicalAccess` with a full named logical-expression environment and the exact active instruction domain, with `index_map` as a cached property returning `None` for non-quasi-affine indices.
2. Define one shared `Layout` protocol supporting expression/parameter mapping, dependency collection, persistent hashing, named-space alignment, pullback, validation, access lowering, allocation, and an optional runtime interface. Note that `pullback` requires a reindexing that is both a single-valued named map and convertible to per-axis Pymbolic expressions, because terminal expressions are Pymbolic.
3. Add reusable `PwAff` utilities for named-space alignment, totality, declared ranges, singleton values, and finite ranges. Do not put injectivity proving in `validate`.
4. Define the **flat** layout union and the wrapper-ordering rules:

   ```python
   ArrayLayout = (
       LinearLayout | RectangularLayout | ImageLayout
       | VectorLayout | SeparateLayout | InstancedLayout
   )
   ```

   Ordering, outermost first: at most one `InstancedLayout` (outermost if present), then at most one `SeparateLayout`, then at most one `VectorLayout`, then exactly one terminal. `ImageLayout` may appear neither beneath `InstancedLayout` nor beneath `VectorLayout`. Check these in one pass in `validate` and assert them in the factories. Do not encode legality with generic `TypeVar`s and derived aliases.
5. Define `InstanceScope` and `StorageKind` as **derived** properties of a layout, not stored fields. Define `StorageReference` carrying object name, derived kind, compile-time `object_key`, and derived instance scope — and no instance or epoch key, since every access is current-instance by invariant.
6. Add lowered coordinate variants: `LinearCoordinate`, `ImageCoordinate`, and `VectorCoordinate` with `ScalarLane` / `WholeVector` selections. There is no footprint type.
7. Implement trivial frozen records for all six layouts. `VectorLayout` stores `lane_expr: namedisl.PwAff`; `SeparateLayout` stores `selector_exprs: tuple[namedisl.PwAff, ...]`; `InstancedLayout` stores an ordered `instances: tuple[InstanceSpec, ...]` with at most one `HardwareInstance` (first) and at most one `SequentialInstance`. No wrapper removes or renumbers dimensions.
8. Implement public construction functions: `make_linear_layout`, `make_rectangular_layout`, `make_c_layout`, `make_f_layout`, `make_strided_layout`, `make_image_layout`, `make_vector_layout`, `make_separate_layout`, `make_instanced_layout`. Factories perform expression parsing, compatibility projection construction, named-space alignment, normalization, and validation; record constructors contain no such logic.
9. Document combined injectivity as a provider contract, and state explicitly that it is load-bearing for race-analysis soundness, not only for allocation.
10. Make `LinearLayout.size` required and an `ArithmeticExpression`. There is no `PhysicalStorageDomain` type.
11. Add `PhysicalAllocation`/`PhysicalStorageObject` with `object_key`, derived `kind`, `element_extent`, optional `image_shape`, and `alignment`. No stored `instance_scope`. Reject nonuniform per-instance allocation.
12. Add `RuntimeArrayInterface` carrying physical dimensions, strides, byte size, alignment, and parameter equations.
13. Add a target hook for vector ABI size/alignment, including three-vector padding. No swizzle-capability hook.

### Suggested location

A new `loopy/kernel/layout.py`. `array.py` is already large.

### Tests

- construction, copying, equality, hashing, and mapping for every layout;
- dependency collection;
- rejection of every illegal wrapper combination: nested instanced layouts, a scope inside a representation wrapper, repeated vector or separate wrappers, a scoped image, an image beneath a vector;
- full-environment lowering without positional axis removal;
- totality, range, and named-space alignment checks for piecewise components;
- valid noninjective components whose combined map is injective, such as `(floor(i/4), i mod 4)`;
- confirmation that `validate` does not claim to prove injectivity;
- trivial record construction and `make_*_layout` normalization;
- C/F/fixed-stride expression and structural extent derivation, including origins and padding;
- `LinearLayout` rejection without a size;
- vector lowering with piecewise lane expressions;
- scalar-lane singleton analysis and identity whole-vector analysis;
- separate selector lowering;
- derived `StorageKind`/`InstanceScope` for each layout shape.

## Phase 4: The `Array` value and composition refactor

### Objectives

Introduce `Array` as the canonical `(shape, layout)` value, move array holders from `is-a` to `has-a`, and make layouts canonical while keeping legacy entry points operational.

### Tasks

1. Define `Array` as a frozen dataclass with exactly `shape` and `layout`. Do **not** add `dtype`, `alignment`, `for_atomic`, `offset`, or `tags`; those stay on the holder. Guard this in review — an `Array` that accumulates fields is `ArrayBase` renamed.
2. Change `ArrayArg` and `TemporaryVariable` to hold `array: Array | None`, where `None` means “still legacy”. Add deprecated forwarding properties for `shape`, `dim_tags`, and `offset`.
3. Implement the `copy()` kwarg split routing array-valued kwargs into `array`.
4. Replace the 67 `isinstance(x, ArrayBase)` sites with a single helper (`get_array(x) -> Array | None` or a `HoldsArray` protocol). `ArrayBase` appears zero times in `test/`, `examples/`, and `doc/`, so this is not a documented public break.
5. Decide tag ownership: tags live on the holder; `Array` is an untagged value. Resolve the current double inheritance from `Taggable` via `ArrayBase` and `KernelArgument`.
6. Collapse `ImageArg` and `ConstantArg` from classes into deprecated factory functions alongside the existing `GlobalArg` function, which already exists precisely to prevent `isinstance` use. An image argument becomes an `ArrayArg` with an `ImageLayout` terminal; a constant argument becomes a read-only `ArrayArg` whose derived storage kind is `CONSTANT_BUFFER`. Remove `min_target_axes`/`max_target_axes`; target-axis count is `ImageLayout` arity.
7. Translate legacy constructor inputs in a compatibility conversion function: `order="C"` → `make_c_layout`; `order="F"` → `make_f_layout`; fixed strides → `make_strided_layout`; `offset` → terminal expression; image target axes → `make_image_layout`; `vec` → `make_vector_layout`; `sep` → `make_separate_layout`; `base_indices` → rectangular origins.
8. Preserve a computed `.dim_tags` compatibility view only for exactly representable layouts; support `copy(dim_tags=...)` as a deprecated replacement-layout operation.
9. Rewrite `tag_array_axes` to parse legacy tags and install layouts. Add `set_array_layout` as the direct new API, and new `tag_array_axes` shorthands for instance axes and image coordinates.
10. Remove `storage_shape`, `base_indices`, and `offset` from canonical equality/hashing.
11. Correct the existing Fortran-layout documentation from row-major to column-major.

### Vector conversion details

1. Build the named projection `PwAff` for the legacy vector axis and prove that its range is exactly the zero-based compile-time-constant interval the tag requires.
2. Install `make_vector_layout(lane_expr, length, child)` without removing or renumbering any logical dimension.
3. Preserve existing whole-vector behavior for the currently vectorized iname as an identity lane permutation.
4. Query the target for physical vector size and alignment.

Note the accepted regression: a scalar vector-axis index that is compile-time constant only after unrolling, and not quasi-affine in the domain, is now rejected. No constant-evaluation fallback is retained.

### Separate conversion details

1. Convert each selected axis to a named projection `PwAff` and install `make_separate_layout(selector_exprs, child)`.
2. Do **not** materialize during preprocessing. `SeparateLayout` survives to code generation; see Phase 7.
3. Retire `ArrayArg._separation_info` as IR state. The selector-tuple-to-name mapping becomes lowering output produced from `physical_allocation()`, reusing the existing deterministic naming scheme so generated ABIs do not change.
4. Require the joint selector range to be a parameter-independent finite constant Cartesian product, enumerate tuples lexicographically, and require selector values to be compile-time known at each scalar access.
5. Permit `make_separate_layout(..., child=make_vector_layout(...))`; the wrapper-ordering rules exclude the reverse and repeats.

### Tests

- old tag strings produce equivalent layout expressions;
- `tag_array_axes` retains C/F/fixed-stride behavior;
- vector and separate compatibility projections;
- `ImageArg`/`ConstantArg` factory functions produce the expected layouts and derived storage kinds;
- deprecated `.shape`/`.dim_tags` forwarding, including the rectangular-shape error on a triangular array;
- `copy()` kwarg splitting, including round-trip reconstruction through constructors;
- equality/persistent hashing over the nested `Array`;
- unsupported compatibility projection diagnostics;
- user-provided generic layout and size.

## Phase 5: Allocation and runtime-wrapper model

### Objectives

Move allocation and host validation from logical shape to layout-provided physical information.

### Tasks

1. Replace `TemporaryVariable.nbytes = product(shape) * itemsize` with layout allocation queries.
2. Take per-instance extents from declared sizes (`LinearLayout`) or structural derivation (`RectangularLayout`, `ImageLayout`). Never range an address expression. Vector lanes do not multiply object count; separate selectors produce distinct objects; instance and epoch keys do not multiply per-instance extent.
3. Reject nonuniform per-instance allocation requirements.
4. Update C-family temporary declaration sizing.
5. Update base-storage allocation to use per-object element extent, derived storage kind and instance scope, alignment, and dtype. Group by **semantic** comparison of storage requirements, not by a hash-keyed dictionary.
6. Implement rectangular runtime interfaces: expected physical rank, dimensions, strides, allocation size, and parameter equations.
7. Refactor wrapper parameter inference to consume layout-contributed equations. This is a refactor of `loopy/target/execution.py:185-320`: the `_ArgFindingEquation` record, its `order` and `based_on_names` fields, the regrouping, and `solve_affine_equations_for` all stay; only equation *construction* moves into `RuntimeArrayInterface.equations`.
8. Preserve existing singleton-axis and empty-array stride equivalence.
9. Preserve rectangular output allocation.
10. For generic layouts: validate byte size/alignment when provided, require an explicit runtime interface for output allocation, and reject ambiguous parameter inference.
11. Update initializer validation to compare against physical storage representation, not logical shape.

### Primary files

- `loopy/kernel/data.py`
- `loopy/target/execution.py`
- `loopy/target/c/c_execution.py`
- `loopy/target/pyopencl.py`
- `loopy/target/c/__init__.py`

### Tests

- no regression in rectangular runtime shape/stride checks;
- inference of `n`, `m`, and stride parameters from inputs;
- singleton and empty dimensions;
- rectangular output allocation;
- nonrectangular logical shape backed by a rectangular physical array;
- generic layout with a declared size; rejection when the size is missing;
- missing/ambiguous generic output allocation errors;
- vector physical size and alignment without double-counting lanes;
- lane-correlated child coordinates such as `(floor(i/4), i mod 4)`;
- one allocation per separate selector fiber;
- rejection of nonuniform per-instance allocation;
- base-storage grouping unaffected by hash/equality divergence.

## Phase 6: Explicit storage instances and minimum callable support

### Objectives

Add `make_storage_instances_explicit` as an opt-in translation-unit transform. Do not enable automatic preprocessing conversion until the code-generation, callable, race, and liveness consumers in later phases are ready.

### Tasks

1. Replace implicit `max(AddressSpace)` joins with an explicit legacy scope lattice helper. Do **not** add a new `AddressSpace` value; the new representation is discriminated by the presence of an `Array`, and instance-explicitness is recorded in `KernelState`.
2. Implement a two-copy candidate-scope collision query that can test hypothetical private/local/global instance axes without mutating the kernel, for sound temporary-scope inference before the transform.
3. Implement `make_storage_instances_explicit` at translation-unit scope.
4. Convert old global and constant arrays to `Array`s with global linear/image/separate/vector layouts.
5. Convert old local arrays: add canonically named group dimensions to the logical shape, rewrite all accesses through named maps carrying current group coordinates, and wrap the child layout in `make_instanced_layout` with a `WORKGROUP` `HardwareInstance`.
6. Convert old private arrays: add canonically named group and item dimensions, rewrite accesses, and wrap with a `WORK_ITEM` `HardwareInstance`.
7. Adapt ILP realization to use a `SequentialInstance` for necessarily sequential private axes.
8. Rewrite accesses in instructions and predicates, substitutions, callable `SubArrayRef`s, and initializer/access metadata.
9. Add the minimum callable support the transform needs: specialize callable descriptors first, preserve shape/layout metadata on `SubArrayRef`s, rewrite caller and specialized callee consistently, and reject unsupported instanced callable cases until Phase 8.
10. Validate current-instance and current-epoch access constraints and uniform per-instance allocation polyhedrally.
11. Make the transform idempotent and add an opt-in validator for tests.

### Preprocessing order

Final target order once automatic conversion is enabled:

1. normalize legacy shapes/layouts into `Array`s;
2. infer `auto` shapes and default layouts;
3. realize ILP/sequential-instance semantics;
4. infer temporary scope with candidate-scope collision analysis;
5. specialize callable descriptors and callable kernels;
6. make storage instances explicit across all specialized kernels and their call sites;
7. validate invariants;
8. run target preprocessing;
9. mark preprocessed.

This ordering is mandatory: making instances explicit adds named dimensions and mappings, so independently transforming unspecialized callers and callees is not supported.

Note that separate-layout materialization no longer appears in this list. Late lowering removes it from preprocessing entirely, which is one of its main benefits: no pass rewrites the kernel behind the analyses' backs.

### Tests

- global/constant/local/private conversion;
- exact added shape dimensions and constraints;
- named access-map rewriting without dependence on tuple-prefix positions;
- aligned, total, and in-range instance projection expressions;
- idempotence;
- missing hardware axes;
- nonzero hardware iname bases;
- rejection of noncurrent group/item access;
- rejection of unprovable current-instance access;
- sequential-instance legality and sequentiality;
- legacy generated-code equivalence for representative kernels.

## Phase 7: Layout-driven code generation

### Objectives

Remove code generation's dependence on dim tags, old address spaces, and array subclasses for access syntax.

### Tasks

1. Replace `get_access_info` with layout lowering from `LogicalAccess`, preserving named expressions and the exact instruction-domain-to-logical-index map.
2. Compose every layout `PwAff` with the access map and active code-generation domain.
3. Update C-family expression and instruction lowering to dispatch on storage references and lowered coordinate variants.
4. Move image coordinate generation into `ImageLayout`; stop reconstructing it from the original index tuple.
5. For scalar accesses, require vector lanes and separate selectors to reduce to compile-time singleton integers.
6. Represent whole-vector access explicitly and prove child storage/coordinate invariance plus an identity lane permutation. Reject runtime-dependent or unresolved piecewise lane mappings.
7. **Implement late separate lowering.** Generate one physical argument/declaration per selector tuple from `physical_allocation()` at signature-construction time; resolve each access's selector to a compile-time tuple and emit a reference to the corresponding object. Expand runtime wrappers to present one host argument per physical object using the existing deterministic naming. Verify that generated ABIs match the current early-materialization output for legacy `sep` kernels.
8. Generate declarations from the derived storage kind: global pointer/argument, constant pointer/declaration, local/shared declaration, private automatic declaration, image object.
9. Update OpenCL atomic and volatile qualifiers to query the derived storage kind.
10. Update CUDA shared/private/global/constant declaration handling.
11. Update ISPC private-lane duplication to use explicit instanced layouts.
12. Define `IndexOfCallable` only for layouts with meaningful linear coordinates; reject others clearly.
13. Remove offset application from target code.

### Primary files

- `loopy/kernel/layout.py`
- `loopy/codegen/instruction.py`
- `loopy/target/c/codegen/expression.py`
- `loopy/target/c/__init__.py`
- `loopy/target/opencl.py`
- `loopy/target/cuda.py`
- `loopy/target/ispc.py`
- `loopy/library/function.py`

### Tests

- generated C/OpenCL/CUDA for each storage kind, including `__constant`;
- image access coordinates;
- scalar lane expressions that become singleton constants only after domain restriction;
- piecewise lane expressions with resolved and unresolved branch cases;
- rejection of runtime-dependent lanes and selectors;
- rejection of a whole-vector access whose lane permutation is not the identity;
- late-lowered separate arrays: generated signature, per-access object selection, and ABI equivalence with the current output;
- local/shared and private declarations;
- atomic and volatile accesses;
- unsupported CUDA image diagnostics;
- `indexof` rejection on non-linear layouts.

## Phase 8: Callable descriptors and subarray composition

### Objectives

Preserve polyhedral shape/layout semantics across callable-kernel boundaries.

### Tasks

1. Change `ArrayArgDescriptor` to hold one `Array` and drop `address_space`. Introduce a separate unresolved specialization state where necessary; do not use `shape=None` or `layout=None`.
2. Implement descriptor mapping and dependency collection over the held `Array`.
3. Rewrite `get_arg_descriptor_for_expression`:
   - obtain the exact swept-iname domain;
   - construct the named map into source logical indices;
   - set the callee shape to the swept domain and **check** containment in the preimage of the source shape, rather than intersecting the two;
   - pull back every layout component, including lane, selector, and instance `PwAff`s;
   - preserve correlated domains and translate parameter namespaces;
   - fix nonswept storage-instance dimensions to current values;
   - require an injective reindexing map as the `pullback` precondition.
4. Replace the minimum Phase 6 callable adapter with full named layout pullback.
5. Support both selector specialization (fixing a fiber) and selector sweeping (passing all objects) at a call boundary.
6. Validate instanced-layout compatibility at call boundaries.
7. Update nested callable inference.
8. Add third-party migration helpers: `descr.array.rectangular_shape()`, `descr.array.linear_strides()`, `descr.array.layout`. Update bundled external-call examples.

### Tests

- rectangular subarrays;
- triangular and diamond swept domains;
- correlated swept inames;
- composed strided layouts and piecewise vector lanes;
- parameter renaming in layout `PwAff`s;
- repeated-element/noninjective actual rejection;
- an out-of-bounds swept domain rejected rather than silently narrowed;
- scalar selector fixed at a call and swept selectors spanning multiple objects;
- local view passed within current group;
- private view passed within current item;
- scope-changing call rejection;
- nested callable specialization;
- output and in/out arguments;
- symbolic parameter translation;
- third-party `with_descrs` compatibility helper use.

## Phase 9: Race, ordering, and barrier analysis

### Objectives

Move race-related analysis onto logical index sets, with layout injectivity as the soundness premise.

This phase is substantially smaller than an earlier draft made it look. There is no physical-coordinate composition, no footprint model, and no atomic-granularity model. The work is to replace the syntactic iname-occurrence test with a real two-copy collision query over logical indices, and to delete the address-space-specific branching that explicit instance axes make unnecessary.

### Tasks

1. Add a shared builder for `execution domain -> array name + logical index set`.
2. Canonicalize group/item expressions by hardware tag, including bounds and nonzero bases.
3. Replace syntactic write-race checks with a two-copy collision query: duplicate execution coordinates, require a difference in at least one relevant concurrent coordinate, require the same array name, require overlapping schedule-derived live ranges and intersecting logical index sets, and test nonemptiness.
4. Reuse and generalize the candidate-scope query introduced in Phase 6; do not maintain a second inference algorithm.
5. Update schedule-time `WriteRaceChecker`, access-range overlap, and variable-access ordering to consume logical index sets and drop local/global branching.
6. Treat unknown or non-affine logical index expressions conservatively.
7. For `base_storage` aliases, conservatively assume overlap between distinct array names sharing base storage whenever their live ranges overlap. Do not compose layouts into a common physical coordinate.
8. Permit sequential-instance reuse only after sequentiality and live-range nonoverlap are proved.
9. Remove `_is_racing_iname_tag` and address-space-specific race logic once unused.

### Tests

Self-race cases:

- `a[i]` → no race;
- `a[2*i]` → no race over an appropriate domain;
- `a[i % 2]` → race;
- `a[i-i]` → race;
- `a[i+j]` over multiple concurrent axes → collision where applicable;
- extent-one concurrent axis → no race from that axis.

Storage-instance cases (all falling out of logical index disjointness):

- global access from different groups/items → may alias;
- local access from different groups → no alias;
- local access from same group/different items → may alias;
- private access from different items → no alias;
- sequential epochs → no concurrent race.

Representation cases:

- distinct scalar lanes of one vector do not overlap;
- a whole-vector access overlaps every constituent lane, because it is a set of logical points;
- distinct separate selectors do not overlap;
- an unknown or non-affine access is conservative.

Ordering/barrier cases:

- disjoint injective accesses require no barrier;
- overlapping base-storage aliases conflict conservatively;
- non-affine unknown access is conservative.

## Phase 10: Generated-subkernel liveness and storage queries

### Objectives

Replace semantic address-space queries used by scheduling and host allocation, and integrate liveness across generated-subkernel boundaries.

### Tasks

1. Replace or redefine `LoopKernel.global_var_names`, `LoopKernel.local_var_names`, and `LoopKernel.local_mem_use` in terms of derived storage kind and instance scope.
2. Replace `_should_temp_var_be_passed` with a liveness query combined with storage kind, instance scope, and ownership.
3. Compute whether each value is live across each generated-subkernel boundary at the finest available program-point granularity.
4. Update generated-subkernel checks: a live `GLOBAL_BUFFER`/`CONSTANT_BUFFER` value may be passed across launches; a live `LOCAL_MEMORY` or `PRIVATE_MEMORY` value cannot without explicit save/reload; sequential-instance reuse requires nonoverlapping epoch live ranges.
5. Update PyOpenCL host allocation and release of global temporaries from computed live ranges.
6. Update base-storage grouping and nested-base-storage checks.
7. Update save/reload and barrier-related users in core paths.
8. Update local-memory statistics to query layouts.
9. Once Phases 7–10 pass their compatibility suites, enable automatic `make_storage_instances_explicit` in preprocessing and activate the hard pre-codegen invariant check.

### Tests

- global temporary across a global barrier;
- local/private temporary rejected across a generated launch boundary;
- base storage across subkernels;
- initialized persistent constants;
- local-memory accounting;
- sequential reuse accepted for disjoint epoch live ranges and rejected for overlap.

## Phase 11: Persistence, diagnostics, and cleanup

### Tasks

1. Update Python reproducer generation for named sets, `Array`s, and layouts.
2. Verify pickle and persistent-cache round trips.
3. Add concise `Array` and layout stringification, preserving the Phase 1 zero-based-box rule.
4. Add diagnostics for:
   - nonrectangular operation requiring a rectangular shape;
   - missing `LinearLayout` size;
   - noncurrent instanced access;
   - unresolved `Array` before code generation;
   - misaligned, non-total, or out-of-range layout `PwAff`;
   - wrapper-ordering violations;
   - dynamic scalar lane or selector, and non-identity whole-vector lane permutation;
   - nonuniform per-instance allocation;
   - unsupported runtime output allocation;
   - noninjective built-in legacy layout.
5. Remove stored dim tags, `ArrayBase`, and obsolete helpers after compatibility coverage is complete.
6. Remove canonical `base_indices`, `storage_shape`, and `offset` handling.
7. Audit equality and hashing so all semantic `ArrayArg` and `TemporaryVariable` fields agree.

## Deferred: transformation migration

A detailed plan for migrating `loopy/transform/` is deliberately not included. The design should solidify first.

What is known now, recorded so it is not rediscovered later: there are **94 references to `.shape` or `dim_tags` across 11 transform modules** — `padding.py` (27), `data.py` (19), `privatize.py` (11), `concatenate.py` (7), `batch.py` (6), `pack_and_unpack_args.py` (6), `diff.py` (6), `precompute.py` (5), `save.py` (3), `callable.py` (3), `buffer.py` (1). Several construct shape tuples element-wise: `padding.py` reorders axes and strides, `batch.py` prepends a batch axis, `concatenate.py` sums extents. `padding.py` is entirely about physical layout and is the transform most affected by the layout model.

The deprecated forwarding properties introduced in Phase 4 keep these modules working on rectangular arrays throughout the phases above. They will fail, by design and with a targeted error, on a nonrectangular array. That is the intended interim behavior; a transform-by-transform plan can be written once the layout API has stopped moving.

## Documentation plan

### `doc/ref_kernel.rst`

Add sections for:

- named-set logical shapes, `auto` as a construction-time request, and the prohibition on `None` shapes;
- shape parameters and zero-dimensional scalars;
- the `Array` value and why arguments and temporaries hold rather than inherit it;
- layouts versus logical shapes;
- the shared `Layout` interface, full named logical-index environment, combined layout map, wrapper ordering, and `make_*_layout` factories;
- linear and rectangular layouts, and the rule that indexing and sizing expressions are supplied independently;
- image layouts and the current absence of texel-channel access;
- vector lanes, identity whole-vector access, and the absence of general swizzles;
- separate layouts and late lowering;
- instanced layouts covering workgroup, work-item, and sequential reuse;
- combined-map injectivity, why it is load-bearing for race analysis, and trusted custom layouts;
- explicit storage instances and why the instance axes are logical;
- physical storage objects, derived kinds and scopes, ownership, reuse epochs, and per-instance extent;
- shape equality versus hashing, and the rule that no correctness decision may depend on hash identity;
- legacy `tag_array_axes` compatibility.

Correct the current Fortran-layout description to column-major.

### `doc/ref_call.rst`

Document polyhedral array descriptors, subarray shape/layout pullback through named maps, the containment check for swept domains, callable reindexing injectivity and specialization, current-group/current-item restrictions, and third-party callable migration helpers.

### `doc/ref_internals.rst`

Document preprocessing order, instance-explicitness invariants, layout-driven lowering, late separate lowering, runtime interface equations, allocation by storage fiber with declared extents, logical-level race analysis and its injectivity premise, and generated-subkernel liveness rules.

### Tutorial/examples

- triangular temporary;
- diamond-shaped tile;
- triangular logical shape in rectangular physical storage;
- packed user-sized linear layout with an explicit size expression;
- local and private arrays with explicit instance axes;
- packed vector layout using `(floor(i/w), i mod w)`;
- separate arrays of vectors;
- callable receiving a nonrectangular subarray.

## Test organization

Prefer focused modules rather than adding every case to `test_loopy.py`:

- `test/test_array_shapes.py`;
- `test/test_array_layouts.py`;
- `test/test_storage_instances.py`;
- existing `test/test_callables.py` for call-boundary cases;
- target-specific execution tests for wrappers and declarations.

Use generated-code checks sparingly. Prefer semantic inspection of normalized kernels and layouts, with execution tests for representative OpenCL/CUDA-capable environments.

## Suggested pull-request sequence

1. Baseline tests and named-set utilities.
2. Canonical named-set shapes and polyhedral bounds checking.
3. Exact `auto` shape inference with bounding-box default layouts.
4. Layout framework: flat union, wrapper ordering, factories, allocation records, linear/rectangular layouts.
5. `Array` value and the composition refactor, including `ImageArg`/`ConstantArg` collapse.
6. Piecewise-affine vector/separate components and legacy tag conversion.
7. Runtime rectangular interface and allocation migration.
8. Candidate-scope race query plus opt-in instance-explicitation and minimum callable rewriting.
9. Layout-driven code generation, including late separate lowering, with dual-path compatibility.
10. Full callable descriptor composition.
11. Logical-level race/order/barrier analysis.
12. Generated-subkernel liveness/storage-query migration and activation of automatic instance-explicitation.
13. Cleanup, deprecations, documentation, and examples.

Each pull request should preserve a runnable tree and add its own compatibility tests. Avoid a single flag day that changes shapes, layouts, code generation, and races simultaneously.

## Completion criteria

- `None` is never an array shape or resolved callable-descriptor shape, and resolved shapes are named sets.
- Every array holds an `Array`; `ArrayBase` inheritance is gone, and `ImageArg`/`ConstantArg` are factory functions.
- `Array` contains exactly a shape and a layout.
- Triangular, diamond, and union-shaped arrays pass bounds checking.
- Storage instances are explicit logical dimensions before code generation, and noncurrent instanced accesses are rejected.
- Layouts, not dim tags, lower accesses and determine allocation.
- All layout nodes receive one full named logical environment and contribute to one combined map.
- Wrapper ordering is enforced by `validate` and the factories, without generic type parameters and without assigning exclusive ownership of logical axes.
- `StorageKind` and `InstanceScope` are derived, never stored.
- Generic extents are declared; no extent is obtained by ranging an address expression.
- Vector lanes are piecewise quasi-affine, with static scalar lowering and identity-only whole-vector lowering.
- Separate layouts are lowered late, and their generated ABI matches the current output for legacy `sep` kernels.
- Race analysis works on logical index sets only, with layout injectivity as its stated soundness premise.
- Rectangular wrappers retain physical shape/stride checks and parameter inference.
- Callable descriptors preserve polyhedral domains and composed layouts.
- Generated-subkernel passing and allocation decisions use liveness plus derived storage kind and scope.
- No correctness decision depends on hash-based identity of a shape or layout.
- Legacy tuple shapes and `tag_array_axes` continue to work through normalization.
- Documentation and persistence formats cover the new model.
