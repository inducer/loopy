# Polyhedral Array Implementation Plan

## Scope

This plan implements the design in `ARRAY-DESIGN.md` in staged, reviewable changes. It covers Loopy core, public compatibility entry points, documentation, and correctness tests. It does not attempt to migrate all existing transformations in the initial effort.

The final code-generation invariant is:

```text
all arrays have named-set shapes, resolved layouts, and UNIVERSAL address space
```

## Guiding implementation rules

- Keep one canonical source of truth for shape and layout.
- Accept legacy forms at API boundaries and normalize immediately through dedicated constructor functions.
- Prefer `make_xyz` functions for conversion, normalization, validation, and other construction policy. Keep canonical records' constructors trivial and preferably dataclass-generated; backward-compatible class constructors should delegate to helpers rather than accumulating logic.
- Do not retain `tuple | namedisl.Set` as the resolved shape representation.
- Do not infer generic physical allocation size from logical shape.
- Trust explicitly custom layout injectivity initially. C/F and other layouts that are injective by construction enforce their structural preconditions; arbitrary symbolic `StridedLayout` values use the trusted custom-layout path rather than requiring a solver proof.
- Preserve rectangular wrapper validation and parameter inference.
- Prefer conservative rejection over unsound race, alias, or lifetime reasoning.
- Add explicit pre-codegen invariants so partially migrated kernels fail early.

## Phase 0: Baseline and decision prototypes

### Objectives

- Record current behavior before changing representation.
- Prototype named-set hashing/copying and parameter substitution.
- Confirm whether `.shape` can change directly or needs an `.index_set` transition.

### Tasks

1. Add characterization tests for:
   - tuple and string shape construction, including rejection of tuple entries equal to `None`;
   - scalar arrays with `shape=()`;
   - `shape=None` and `shape=auto`;
   - C/F/fixed-stride layouts;
   - vector and separate dim tags;
   - shape/stride runtime checks and parameter inference;
   - local/private/global temporary declarations;
   - image access;
   - subarray callable descriptors;
   - generated-subkernel temporary passing;
   - current race and barrier behavior.
2. Search external-facing examples and tests for direct tuple operations on `.shape`.
3. Prototype stable equality and persistent hashing for aligned `namedisl.Set`s.
4. Prototype substitution/renaming of shape parameters for callable descriptor translation.
5. Decide the migration spelling using explicit go/no-go criteria:
   - prefer `.shape` immediately when all in-tree core consumers can migrate atomically, persistence is stable, and compatibility impact is limited to documented direct tuple introspection;
   - use canonical `.index_set` plus a deprecated rectangular `.shape` view for one release if core consumers or essential third-party extension points cannot migrate atomically.
6. Prototype and settle before publishing layout classes:
   - the concrete physical-storage-domain type;
   - target vector ABI hooks;
   - generic runtime-interface minimums;
   - the closed layout child-type states and public `make_*_layout` factory signatures.

### Primary files

- `loopy/kernel/array.py`
- `loopy/kernel/data.py`
- `loopy/tools.py`
- `test/test_loopy.py`
- `test/test_callables.py`
- target runtime tests

### Exit criteria

- Existing behavior is covered by focused tests.
- Named-set values can be copied, mapped, compared, pickled, and persistently hashed deterministically.
- The `.shape` migration strategy is chosen using the stated go/no-go criteria.
- Blocking layout API decisions are settled before Phase 3.

## Phase 1: Canonical shape utilities

### Objectives

Introduce named-set shapes without yet removing all legacy layout code.

### Tasks

1. Change the canonical shape type alias to:

   ```python
   ArrayShape = namedisl.Set | type[auto] | None
   ```

2. Add shape-construction and conversion helpers:
   - legacy tuple/string to named set, rejecting tuple entries equal to `None`;
   - zero-dimensional point construction;
   - rank-known universe construction;
   - named-axis normalization;
   - shape parameter dependency extraction.
3. Add `get_default_shape_axis_name(i)` alongside `get_access_map_storage_names`. It returns `_lpy_s{i}` for zero-based axis `i`; `_lpy_` is protected, so it performs no collision avoidance. Make `get_access_map_storage_names` use this helper.
4. Add shape queries:
   - `num_axes` as an attribute;
   - logical axis names;
   - `rectangular_shape()`;
   - scalar-shape predicate;
   - empty-shape predicate.
5. Route legacy shape inputs through the construction helper. Keep `ArrayBase.__init__` trivial except for unavoidable backward-compatible delegation, and do not merge this change until every in-tree core `.shape` consumer is migrated in the same pull request or the `.index_set` transition adapter is in place.
6. Replace `dim_names` canonical storage with set dimension names.
7. Update `ArrayBase`:
   - equality;
   - persistent hashing;
   - stringification, using `namedisl.Set.is_box` to print boxes in NumPy shape notation and named-set notation otherwise;
   - `num_axes`;
   - expression/parameter mapping;
   - scalar detection.
8. Update `ArrayArg` and `TemporaryVariable` annotations and compatibility construction functions.
9. Keep compatibility parsing for tuple/string shape inputs.
10. Define targeted errors for operations requiring a rectangular shape.

### Primary files

- `loopy/typing.py`
- `loopy/kernel/array.py`
- `loopy/kernel/data.py`
- `loopy/symbolic.py`
- `loopy/tools.py`

### Tests

- tuple-to-set conversion;
- deterministic `_lpy_s{i}` dimension naming and protected-namespace behavior;
- rejection of `None` tuple entries;
- zero-dimensional point/scalar semantics;
- empty sets;
- nonzero lower bounds;
- triangular, diamond, and union shapes;
- rectangular extraction success and failure;
- box stringification in NumPy shape notation and non-box set stringification;
- equality/hash stability under parameter alignment;
- pickle and reproducer round trips.

### Exit criteria

Resolved array shapes are named sets throughout the modified core paths. No new code branches on tuple versus set.

## Phase 2: Polyhedral bounds and shape inference

### Objectives

Make shape-consuming analysis uniformly polyhedral.

### Tasks

1. Rewrite `_AccessCheckMapper.map_subscript` to:
   - build the access range;
   - align parameters and output dimensions with the array shape;
   - check subset containment directly.
2. Update shape-parameter validation to inspect set parameters and require integral read-only `ValueArg`s for externally visible arrays.
3. Change `find_shapes_of_vars` to return access-range sets rather than base-index/extent tuples, and fail if any relevant access is unanalyzable instead of unioning only the successful accesses.
4. Change `determine_shapes_of_temporaries` to retain exact access unions.
5. Change `guess_var_shape` to return named sets.
6. Remove all bounding-box shape inference and its options; inference either retains the exact access union or fails.
7. Migrate scalar-array checks from `shape == ()` to a helper.
8. Deprecate or remove `TemporaryVariable.base_indices` from canonical behavior.

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
- non-quasi-affine inference failure;
- scalar and empty arrays.

### Exit criteria

All core bounds and inference operations consume named sets directly. No internal shape-inference analysis reconstructs a bounding box.

## Phase 3: Layout object framework

### Objectives

Introduce immutable layout values and access-lowering protocols while retaining adapters for current dim tags.

### Tasks

1. Define one shared `Layout` protocol supporting:
   - expression/parameter mapping;
   - dependency collection;
   - persistent hashing;
   - validation against the logical shape;
   - access lowering;
   - physical allocation requirements;
   - an optional runtime interface description.
2. Define closed layout-state aliases and narrow child types:
   - `ElementTerminalLayout = LinearLayout | RectangularLayout`;
   - `TerminalLayout = ElementTerminalLayout | ImageLayout`;
   - generic `VectorLayout[VectorChildT]`, where `VectorChildT` is covariantly bounded by `ElementTerminalLayout | ImageLayout`;
   - `ElementVectorLayout`, `ImageVectorLayout`, and their `AnyVectorLayout` union;
   - `SeparateChildLayout = TerminalLayout | AnyVectorLayout` and covariant generic `SeparateLayout[SeparateChildT]`;
   - `ElementRepresentationLayout`, excluding images, image-backed vectors, and separated variants of either;
   - `InamePrivateLayout.child: ElementRepresentationLayout`;
   - `InstanceScopedChildLayout = ElementRepresentationLayout | InamePrivateLayout`;
   - `LocalLayout.child` and `PrivateLayout.child` use `InstanceScopedChildLayout`;
   - `ArrayLayout` is the closed union of all legal root states and is distinct from the shared `Layout` protocol.
3. Define one `StorageReference` carrying object name, storage kind, instance scope, and lifetime, plus lowered coordinate variants for linear, vector, and image coordinates. Separate lowering selects a storage reference rather than nesting competing storage names.
4. Implement trivial frozen records for terminal layouts:
   - `LinearLayout(expr, size)`;
   - `RectangularLayout(axes, base_offset)` with explicit logical axes, origins, extents, and strides;
   - `ImageLayout(axis_exprs, physical_shape)`.
5. Implement trivial frozen records for scope wrappers:
   - `LocalLayout`;
   - `PrivateLayout`;
   - `InamePrivateLayout`.
6. Implement trivial frozen records for representation wrappers:
   - generic `VectorLayout`;
   - generic `SeparateLayout`.
7. Implement public construction functions, including `make_linear_layout`, `make_rectangular_layout`, `make_c_layout`, `make_f_layout`, `make_strided_layout`, `make_image_layout`, and one `make_*_layout` function per wrapper. Factories perform expression parsing, compatibility conversion, normalization, and value-level validation; record constructors contain no such logic.
8. Encode wrapper-order legality in factory signatures and child annotations. This statically excludes nested hardware scopes (including `PrivateLayout` under `PrivateLayout`), scope wrappers inside representation wrappers, `VectorLayout(SeparateLayout(...))`, repeated vector/separate wrappers, and scoped images or image-backed vectors, while permitting an image-backed vector as a root or top-level separate child. Factories additionally reject overlapping consumed axes and malformed values.
9. Define axis-consumption APIs so every wrapper validates and removes its axes before invoking its child.
10. Add a structured `PhysicalAllocation` result describing physical objects, per-object element extent, alignment, instance scope, and lifetime.
11. Add a runtime-interface record carrying physical dimensions, strides, byte size, alignment, and parameter equations.
12. Make generic `LinearLayout` require an explicit size or physical storage domain.
13. Add target hooks for element-backed vector ABI size/alignment; image-backed vectors use image-format channel metadata instead.
14. Expose image texel channel count through the image layout or associated storage metadata so factories and code generation can validate image-backed vectors.

### Suggested location

Initially place public layout definitions in `loopy/kernel/array.py` or a new `loopy/kernel/layout.py`. A separate module is preferable once the API stabilizes because `array.py` is already large.

### Tests

- construction, copying, equality, hashing, and mapping for every layout;
- dependency collection;
- static and runtime rejection of every illegal child-type combination, including nested `PrivateLayout`;
- factory validation of overlapping consumed axes;
- trivial record construction and `make_*_layout` normalization;
- C/F/fixed-stride expression and allocation derivation, including origins and padding;
- generic layout rejection without size;
- element-backed and image-backed vector access lowering;
- image-vector channel-count validation;
- rejection of image-backed vectors beneath scope wrappers;
- separate access lowering, including separate image vectors;
- image coordinates;
- local/private storage identity metadata.

### Exit criteria

The shared layout protocol and closed legal-root union can represent every current core layout feature, and illegal wrapper combinations are excluded by child types and factory signatures, even if code generation still uses compatibility adapters.

## Phase 4: Legacy layout conversion and public APIs

### Objectives

Make layouts canonical while keeping legacy user entry points operational.

### Tasks

1. Change `ArrayBase` to store `layout`, not `dim_tags`, `offset`, `strides`, or `order` as independent canonical state.
2. Translate legacy constructor inputs in a compatibility conversion function:
   - `order="C"` -> `make_c_layout`;
   - `order="F"` -> `make_f_layout`;
   - fixed strides -> `make_strided_layout`;
   - offset -> terminal layout base offset/expression;
   - image target axes -> `make_image_layout`;
   - `vec` -> `make_vector_layout`;
   - `sep` -> `make_separate_layout`.
3. Preserve a computed `.dim_tags` compatibility view only for exactly representable layouts.
4. Support `copy(dim_tags=...)` as a deprecated replacement-layout operation.
5. Rewrite `tag_array_axes` to parse legacy tags and install layouts.
6. Add `set_array_layout` as the direct new API.
7. Add new `tag_array_axes` shorthands or structured arguments for:
   - local group-instance axes;
   - private group/item-instance axes;
   - iname-private axes;
   - image coordinates.
8. Remove `storage_shape`, `base_indices`, and `offset` from canonical equality/hashing and migrate inexpensive legacy input cases.
9. Correct the existing Fortran-layout documentation from row-major to column-major.

### Vector conversion details

For a legacy vector axis:

1. Require the selected axis to be exactly a zero-based compile-time-constant interval, independent of child coordinates.
2. Remove that axis from the child rectangular/linear layout.
3. Install `make_vector_layout(axis, length, child)`.
4. Preserve existing whole-vector behavior for the currently vectorized iname.
5. Query the target for physical vector size and alignment, including three-vector padding.

### Separate conversion details

For legacy separate axes:

1. Install `make_separate_layout(axes, child)` as the semantic representation.
2. Initially materialize physical subarguments during preprocessing using the existing naming/ABI strategy.
3. Record the mapping from selector tuples to materialized argument names as lowering output, not canonical array metadata.
4. Remove selected axes from child shapes/layouts after materialization.
5. Require selector axes to be parameter-independent zero-based constant Cartesian intervals, enumerate tuples lexicographically, and require selector values to be compile-time known at each access.
6. Permit `make_separate_layout(..., child=make_vector_layout(...))`; exclude the reverse order and repeated wrappers through child types, and reject overlapping consumed axes in the factories.

### Tests

- old tag strings produce equivalent layout expressions;
- `tag_array_axes` retains C/F/fixed-stride behavior;
- vector and separate compatibility cases;
- nested separate/vector combinations;
- unsupported compatibility projection diagnostics;
- user-provided generic layout and size.

### Exit criteria

Resolved arrays have one canonical layout. Existing common constructors and `tag_array_axes` still work through conversion.

## Phase 5: Allocation and runtime-wrapper model

### Objectives

Move allocation and host validation from logical shape to layout-provided physical information.

### Tasks

1. Replace `TemporaryVariable.nbytes = product(shape) * itemsize` with layout allocation queries.
2. Update C-family temporary declaration sizing.
3. Update base-storage allocation to use:
   - structured per-object element extent;
   - physical storage kind;
   - storage-instance scope without multiplying launch size;
   - lifetime;
   - alignment and dtype.
4. Implement rectangular runtime interfaces containing:
   - expected physical rank and dimensions;
   - expected strides;
   - allocation size;
   - equations for inferring size parameters.
5. Refactor wrapper parameter inference to consume layout-contributed equations.
6. Preserve existing singleton-axis and empty-array stride equivalence where valid.
7. Preserve rectangular output allocation.
8. For generic layouts:
   - validate byte size/alignment when provided;
   - require explicit runtime interface for output allocation;
   - reject ambiguous parameter inference.
9. Update initializer validation to compare against physical storage representation, not logical shape.

### Primary files

- `loopy/kernel/data.py`
- `loopy/target/execution.py`
- `loopy/target/c/c_execution.py`
- `loopy/target/pyopencl.py`
- `loopy/target/c/__init__.py`
- base-storage allocation code

### Tests

- no regression in rectangular runtime shape/stride checks;
- inference of `n`, `m`, and stride parameters from inputs;
- singleton and empty dimensions;
- rectangular output allocation;
- nonrectangular logical shape backed by a rectangular physical array;
- generic byte-size validation;
- missing/ambiguous generic output allocation errors;
- vector physical size and alignment;
- separate child allocation.

### Exit criteria

No core allocation or wrapper path computes storage as a product of logical shape dimensions.

## Phase 6: Universal address-space transform and minimum callable support

### Objectives

Add `AddressSpace.UNIVERSAL` and an opt-in normalization transform. Do not enable automatic preprocessing conversion until the code-generation, callable, race, and lifetime consumers in later phases are ready.

### Tasks

1. Append `UNIVERSAL` without renumbering existing enum values.
2. Replace implicit `max(AddressSpace)` joins with an explicit legacy scope lattice/helper.
3. Implement a two-copy candidate-scope collision query that can test hypothetical private/local/global instance axes without mutating the kernel. Use it for sound temporary-scope inference before universalization.
4. Implement `to_universal_address_space` at translation-unit scope.
5. Convert old global arrays to universal arrays with global linear/image/separate/vector layouts.
6. Convert old local arrays:
   - add canonical group dimensions to the logical shape;
   - rewrite all accesses with current group coordinates;
   - wrap the child layout in `LocalLayout`.
7. Convert old private arrays:
   - add canonical group and item dimensions;
   - rewrite accesses with current coordinates;
   - wrap in `PrivateLayout`.
8. Adapt ILP realization to use `InamePrivateLayout` for necessarily sequential private axes.
9. Rewrite accesses in:
   - instructions and predicates;
   - substitutions;
   - callable `SubArrayRef`s;
   - initializer/access metadata where applicable.
10. Add the minimum callable support needed by the translation-unit transform:
   - specialize callable descriptors before universalization;
   - preserve shape/layout metadata on `SubArrayRef`s;
   - rewrite caller and specialized callee consistently;
   - reject unsupported scoped callable cases until Phase 8 completes composition support.
11. Validate current-instance access constraints polyhedrally.
12. Make the transform idempotent.
13. Keep automatic preprocessing conversion disabled in this phase; old-address-space code generation remains the default compatibility path.
14. Add an opt-in universal-IR validator for tests and development.

### Preprocessing order

Final target order once automatic conversion is enabled:

1. normalize legacy shapes/layouts;
2. infer `auto` shapes;
3. materialize `SeparateLayout` before callable descriptor construction;
4. realize ILP/iname-private semantics;
5. infer temporary scope with candidate-scope collision analysis;
6. specialize callable descriptors and callable kernels;
7. universalize all specialized kernels and their call sites;
8. validate universal invariants;
9. run target preprocessing;
10. mark preprocessed.

This ordering is mandatory: universalization changes rank by adding instance dimensions, so independently universalizing unspecialized callers and callees is not supported.

### Tests

- global/local/private conversion;
- exact added shape dimensions and constraints;
- access rewriting;
- idempotence;
- missing hardware axes;
- nonzero hardware iname bases;
- rejection of noncurrent group/item access;
- rejection of unprovable current-instance access;
- iname-private legality and sequentiality;
- legacy generated code equivalence for representative kernels.

### Exit criteria

The opt-in transform produces a self-consistent universal translation unit, including supported callable edges. Automatic preprocessing conversion remains gated on the Phase 7-10 consumer migrations.

## Phase 7: Layout-driven code generation

### Objectives

Remove code generation's dependence on dim tags, old address spaces, and array subclasses for access syntax.

### Tasks

1. Replace `get_access_info` with layout lowering.
2. Update C-family expression lowering to dispatch on lowered access variants.
3. Move image coordinate generation into `ImageLayout`; stop reconstructing it from the original index tuple.
4. Implement vector lane and whole-vector lowering from `VectorCoordinate`, including image-backed vectors whose selected axis maps to texel channels.
5. Validate that each image-backed vector length equals the image format's channel count before emission.
6. Lower image-vector lane reads as a whole image read followed by channel selection, and lower whole-vector image accesses as whole-texel operations.
7. Reject every partial image-vector write during code generation. Do not synthesize read-modify-write; only a write proven to cover all channels exactly once may emit an image store, and uncertain coverage fails conservatively with an actionable diagnostic.
8. Consume the storage references selected by early separate materialization.
9. Generate declarations from layout storage kind:
   - global pointer/argument;
   - local/shared declaration;
   - private automatic declaration;
   - image object.
10. Update OpenCL atomic and volatile qualifiers to query physical storage kind.
11. Update CUDA shared/private/global declaration handling.
12. Update ISPC private-lane duplication to use explicit private layouts.
13. Define `IndexOfCallable` only for layouts with meaningful linear coordinates; reject others clearly.
14. Remove offset application from target code.

### Primary files

- `loopy/kernel/array.py` or `loopy/kernel/layout.py`
- `loopy/codegen/instruction.py`
- `loopy/target/c/codegen/expression.py`
- `loopy/target/c/__init__.py`
- `loopy/target/opencl.py`
- `loopy/target/cuda.py`
- `loopy/target/ispc.py`
- `loopy/library/function.py`

### Tests

- generated C/OpenCL/CUDA for each storage kind;
- image access coordinates;
- image-backed `float4` channel-axis reads and whole-vector reads/writes;
- code-generation rejection of scalar-lane and proper-subvector image writes, with no read-modify-write output;
- image-vector channel-count mismatch diagnostics;
- ordinary vector lanes and whole-vector access;
- separate arrays, including separate image vectors;
- local/shared and private declarations;
- atomic and volatile accesses;
- unsupported CUDA image diagnostics;
- `indexof` rejection on non-linear layouts.

### Exit criteria

Target expression and declaration code does not inspect dim tags or old array address spaces.

## Phase 8: Callable descriptors and subarray composition

### Objectives

Preserve polyhedral shape/layout semantics across callable-kernel boundaries.

### Tasks

1. Change `ArrayArgDescriptor` to carry named-set shape and layout.
2. Implement descriptor mapping over shape parameters and layout expressions.
3. Implement descriptor dependency collection from set parameters and layouts.
4. Rewrite `get_arg_descriptor_for_expression`:
   - obtain the exact swept-iname domain;
   - construct the map into source logical indices;
   - compose the source layout;
   - preserve correlated domains;
   - fix nonswept storage-instance dimensions to current values.
5. Replace the minimum Phase 6 callable adapter with full subarray layout composition.
6. Validate local/private compatibility at call boundaries.
7. Update nested callable inference.
8. Enforce the preprocessing order fixed in Phase 6: specialization first, translation-unit universalization second.
9. Add third-party migration helpers:
   - `rectangular_shape()`;
   - `linear_strides()`;
   - direct `.layout` access.
10. Update bundled external-call examples to use helpers.

### Tests

- rectangular subarrays;
- triangular and diamond swept domains;
- correlated swept inames;
- composed strided layouts;
- local view passed within current group;
- private view passed within current item;
- scope-changing call rejection;
- vector/separate child descriptors;
- nested callable specialization;
- output and in/out arguments;
- symbolic parameter translation;
- third-party `with_descrs` compatibility helper use.

### Exit criteria

No callable descriptor collapses a set shape into independent extents or assumes fixed stride tags.

## Phase 9: Race, ordering, and barrier analysis

### Objectives

Use universal logical indices and refined injectivity for all race-related analysis.

### Tasks

1. Add a shared builder for:

   ```text
   execution domain -> universal logical array index
   ```

2. Canonicalize group/item axes by hardware tag, including bounds and nonzero bases.
3. Replace syntactic write-race checks with a two-copy collision query:
   - duplicate execution coordinates;
   - require a difference in at least one relevant concurrent coordinate;
   - require equal universal logical indices;
   - test nonemptiness.
4. Reuse and generalize the candidate-scope query introduced in Phase 6; do not maintain a second inference algorithm.
5. Update schedule-time `WriteRaceChecker` to remove local/global branching.
6. Update access-range overlap and variable-access ordering to use universal logical addresses.
7. Treat unknown/non-affine access conservatively.
8. For base-storage aliases:
   - compare common physical coordinates when available;
   - otherwise preserve conservative overlap.
9. Account for iname-private lifetime nonoverlap.
10. Remove old `_is_racing_iname_tag` and address-space-specific race logic once unused.

### Tests

Self-race cases:

- `a[i]` -> no race;
- `a[2*i]` -> no race over an appropriate domain;
- `a[i % 2]` -> race;
- `a[i-i]` -> race;
- `a[i+j]` over multiple concurrent axes -> collision where applicable;
- extent-one concurrent axis -> no race from that axis.

Storage-instance cases:

- global access from different groups/items -> may alias;
- local access from different groups -> no alias;
- local access from same group/different items -> may alias;
- private access from different items -> no alias;
- iname-private sequential iterations -> no concurrent race.

Ordering/barrier cases:

- disjoint injective accesses require no barrier;
- non-affine unknown access is conservative;
- overlapping base-storage aliases conflict;
- provably disjoint base-storage regions avoid unnecessary conflicts if implemented.

### Exit criteria

Race correctness no longer depends on old address-space enum values or mere occurrence of inames in subscripts.

## Phase 10: Generated-subkernel lifetime and storage queries

### Objectives

Replace semantic address-space queries used by scheduling and host allocation.

### Tasks

1. Replace or redefine:
   - `LoopKernel.global_var_names`;
   - `LoopKernel.local_var_names`;
   - `LoopKernel.local_mem_use`.
2. Replace `_should_temp_var_be_passed` with a lifetime/storage-kind query.
3. Update generated-subkernel checks:
   - persistent global storage may cross launches;
   - local/private storage may not;
   - iname-private storage may not escape its lifetime.
4. Update PyOpenCL host allocation and release of persistent temporaries.
5. Update base-storage grouping and nested-base-storage checks.
6. Update save/reload and barrier-related users in core paths.
7. Update local-memory statistics to query layouts.
8. Once Phases 7-10 pass their compatibility suites, enable automatic `to_universal_address_space` in preprocessing and activate the hard pre-codegen nonuniversal-array check.

### Tests

- global temporary across a global barrier;
- local/private temporary rejected across a generated launch boundary;
- base storage across subkernels;
- initialized persistent constants;
- local-memory accounting;
- iname-private escape rejection.

### Exit criteria

Generated-subkernel lifetime behavior is independent of `AddressSpace.GLOBAL/LOCAL/PRIVATE`, and automatic preprocessing universalization is enabled without falling back to legacy code generation.

## Phase 11: Persistence, diagnostics, and cleanup

### Objectives

Complete the public transition and remove duplicate representations.

### Tasks

1. Update Python reproducer generation for named sets and layouts.
2. Verify pickle and persistent-cache round trips.
3. Add concise layout and shape stringification, preserving the Phase 1 rule that boxes use NumPy shape notation and non-boxes use named-set notation.
4. Add diagnostics for:
   - nonrectangular operation requiring rectangular shape;
   - missing generic allocation size;
   - noncurrent local/private access;
   - unresolved layout before code generation;
   - invalid vector/separate composition;
   - image-vector channel-count mismatch;
   - partial image-vector writes;
   - unsupported runtime output allocation;
   - noninjective built-in legacy layout.
5. Remove stored dim tags and obsolete helpers after compatibility coverage is complete.
6. Remove canonical `base_indices`, `storage_shape`, and `offset` handling.
7. Audit equality and hashing so all semantic `ArrayArg` and `TemporaryVariable` fields agree.

### Exit criteria

There is one canonical representation, all persistent forms are stable, and failures identify the violated layout/shape invariant.

## Documentation plan

### `doc/ref_kernel.rst`

Add sections for:

- named-set logical shapes;
- shape parameters and zero-dimensional scalars;
- layouts versus logical shapes;
- the shared `Layout` interface, legal layout child types, and `make_*_layout` factories;
- linear and rectangular layouts;
- local/private/image layouts;
- vector and separate composition, including image texel channel axes;
- whole-texel image-vector writes and rejection of partial writes;
- iname-private storage;
- refined injectivity;
- universal address space;
- physical allocation requirements;
- legacy `tag_array_axes` compatibility.

Correct the current Fortran-layout description to column-major.

### `doc/ref_call.rst`

Document:

- polyhedral array descriptors;
- subarray shape/layout composition;
- callable specialization;
- current-group/current-item restrictions;
- third-party callable migration helpers.

### `doc/ref_internals.rst`

Document:

- preprocessing order;
- universalization invariants;
- layout-driven lowering;
- runtime interface equations;
- universal race/access maps;
- generated-subkernel lifetime rules.

### Tutorial/examples

Add examples for:

- triangular temporary;
- diamond-shaped tile;
- triangular logical shape in rectangular physical storage;
- packed user-sized linear layout;
- local and private universalized arrays;
- vector layout;
- separate arrays of vectors;
- callable receiving a nonrectangular subarray.

## Test organization

Prefer focused modules rather than adding every case to `test_loopy.py`:

- `test/test_array_shapes.py`;
- `test/test_array_layouts.py`;
- `test/test_universal_addressing.py`;
- existing `test/test_callables.py` for call-boundary cases;
- target-specific execution tests for wrappers and declarations.

Use generated-code checks sparingly. Prefer semantic inspection of normalized kernels/layouts, with execution tests for representative OpenCL/CUDA-capable environments.

## Suggested pull-request sequence

1. Baseline tests and named-set utilities.
2. Canonical named-set shapes and polyhedral bounds checking.
3. Exact `auto` shape inference.
4. Layout protocol and legal-root types, construction factories, structured allocation, and linear/rectangular layouts.
5. Vector and separate wrappers, allocation behavior, and legacy conversion.
6. Runtime rectangular interface and allocation migration.
7. Candidate-scope race query plus opt-in universalization and minimum callable rewriting.
8. Layout-driven code generation with dual-path compatibility.
9. Full callable descriptor composition.
10. Universal race/order/barrier analysis.
11. Generated-subkernel lifetime migration and activation of automatic universalization.
12. Cleanup, deprecations, documentation, and examples.

Each pull request should preserve a runnable tree and add its own compatibility tests. Avoid a single flag day that changes shapes, layouts, code generation, and races simultaneously.

## Completion criteria

The project is complete when:

- resolved `.shape` values are named sets;
- triangular, diamond, and union-shaped arrays pass bounds checking;
- all arrays are universal before code generation;
- local/private instances are explicit logical dimensions;
- noncurrent local/private/iname-private accesses are rejected;
- layouts, not dim tags, lower accesses and determine allocation;
- all layout nodes implement the shared protocol, and illegal child combinations are excluded by types and factories;
- image texel channels may be represented by an image-backed vector layout without permitting scoped images;
- partial image-vector writes are rejected during code generation;
- vector and separate behavior is represented compositionally;
- generic layouts require explicit allocation size;
- rectangular wrappers retain physical shape/stride checks and parameter inference;
- callable descriptors preserve polyhedral domains and composed layouts;
- race checks operate on universal logical indices;
- generated-subkernel lifetime decisions use layout storage semantics;
- legacy tuple shapes and `tag_array_axes` continue to work through normalization;
- documentation and persistence formats cover the new model.
