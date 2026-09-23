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
- Treat injectivity as a layout-provider contract for all layouts, not a responsibility of structural validation. Standard factories document why their constructions satisfy it; arbitrary symbolic/custom layouts remain the provider's responsibility.
- Preserve rectangular wrapper validation and parameter inference.
- Prefer conservative rejection over unsound race, alias, liveness, or storage-reuse reasoning.
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
   - legacy top-level `shape=None` conversion to `auto` only at explicitly supported inference entry points, rejection elsewhere, and direct `shape=auto`;
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
4. Prototype named-space alignment, persistent hashing, parameter substitution, pullback, totality/range checking, and singleton evaluation for `namedisl.PwAff` layout components. Treat injectivity as a documented layout-provider contract rather than a validation query.
5. Decide the migration spelling using explicit go/no-go criteria:
   - prefer `.shape` immediately when all in-tree core consumers can migrate atomically, persistence is stable, and compatibility impact is limited to documented direct tuple introspection;
   - use canonical `.index_set` plus a deprecated rectangular `.shape` view for one release if core consumers or essential third-party extension points cannot migrate atomically.
6. Prototype and settle before publishing layout classes:
   - the concrete physical-storage-domain type;
   - target vector ABI and static-swizzle capability hooks;
   - generic runtime-interface minimums;
   - the closed layout child-type states and public `make_*_layout` factory signatures;
   - the full named logical-index environment and combined layout-map representation;
   - static scalar-lane, selector, and whole-vector swizzle proof APIs.

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
4. Add shape queries:
   - `num_axes` as an attribute;
   - logical axis names;
   - `rectangular_shape()`;
   - scalar-shape predicate;
   - empty-shape predicate.
5. Route legacy shape inputs through the construction helper, ensuring that `None` is never stored in `ArrayBase.shape`. Keep `ArrayBase.__init__` trivial except for unavoidable backward-compatible delegation, and do not merge this change until every in-tree core `.shape` consumer is migrated in the same pull request or the `.index_set` transition adapter is in place.
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
- immediate top-level `shape=None` conversion or targeted rejection, with no stored `None` state;
- zero-dimensional point/scalar semantics;
- empty sets;
- nonzero lower bounds;
- triangular, diamond, and union shapes;
- rectangular extraction success and failure;
- box stringification in NumPy shape notation and non-box set stringification;
- equality/hash stability under parameter alignment;
- pickle and reproducer round trips.

### Exit criteria

Resolved array shapes are named sets throughout the modified core paths. No array or resolved callable descriptor stores `None` as a shape, and no new code branches on tuple versus set.

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
4. Change `determine_shapes_of_temporaries` to retain exact access unions and diagnose an unremoved, access-free `auto` temporary instead of producing an unresolved shape.
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
- access-free `auto` temporary removal before inference or targeted failure;
- non-quasi-affine inference failure;
- scalar and empty arrays.

### Exit criteria

All core bounds and inference operations consume named sets directly. No internal shape-inference analysis reconstructs a bounding box.

## Phase 3: Layout object framework

### Objectives

Introduce immutable layout values and access-lowering protocols while retaining adapters for current dim tags.

### Tasks

1. Define `LogicalAccess` with a full named logical-expression environment and the exact active instruction domain. Implement its instruction-domain-to-logical-index map as a cached property derived from those fields, returning `None` for non-quasi-affine indices.
2. Define one shared `Layout` protocol supporting:
   - Pymbolic expression and named-ISL parameter mapping;
   - dependency collection and persistent hashing;
   - named-space alignment;
   - pullback through a named reindexing map;
   - validation against the exact logical shape;
   - access lowering;
   - shape-aware physical allocation requirements;
   - an optional runtime interface description.
3. Define the combined layout map from the unchanged logical point to storage-object selector, instance key, reuse-epoch key, terminal coordinate, and representation coordinate. Add reusable `PwAff` utilities for named-space alignment, totality, declared ranges, singleton values, and finite ranges; do not put injectivity proving in `validate`.
4. Define closed layout-state aliases and narrow child types:
   - `ElementTerminalLayout = LinearLayout | RectangularLayout`;
   - `TerminalLayout = ElementTerminalLayout | ImageLayout`;
   - generic `VectorLayout[VectorChildT]`, where `VectorChildT` is covariantly bounded by `ElementTerminalLayout | ImageLayout`;
   - `ElementVectorLayout`, `ImageVectorLayout`, and their `AnyVectorLayout` union;
   - `SeparateChildLayout = TerminalLayout | AnyVectorLayout` and covariant generic `SeparateLayout[SeparateChildT]`;
   - `ElementRepresentationLayout`, excluding images, image-backed vectors, and separated variants of either;
   - the existing legal root/scope aliases, distinct from the shared `Layout` protocol.
5. Define `InstanceScope` (`GLOBAL`, `WORKGROUP`, `WORK_ITEM`) and `StorageKind` (`GLOBAL_BUFFER`, `LOCAL_MEMORY`, `PRIVATE_MEMORY`, `IMAGE`). Define one `StorageReference` carrying object name, storage kind, instance key, and reuse-epoch key. Add lowered coordinate variants for linear, image, scalar-vector-lane, and static-swizzle access, plus explicit scalar-lane, whole-vector, and image-texel footprints.
6. Implement trivial frozen records for terminal, scope, and representation layouts. `VectorLayout` stores `lane_expr: namedisl.PwAff`; `SeparateLayout` stores `selector_exprs: tuple[namedisl.PwAff, ...]`; scope mappings store logical `PwAff`s. No wrapper removes or renumbers dimensions.
7. Implement public construction functions, including `make_linear_layout`, `make_rectangular_layout`, `make_c_layout`, `make_f_layout`, `make_strided_layout`, `make_image_layout`, and one `make_*_layout` function per wrapper. Factories perform expression parsing, compatibility projection construction, named-space alignment, normalization, and value-level validation; record constructors contain no such logic.
8. Encode wrapper-order legality in factory signatures and child annotations. This excludes nested hardware scopes, scope wrappers inside representation wrappers, reversed/repeated vector or separate wrappers, and scoped images or image-backed vectors, while permitting an image-backed vector as a root or top-level separate child. Do not reject shared dependencies among lane, selector, and child-coordinate expressions.
9. Document combined injectivity as a provider contract for every layout. Explain why standard factory constructions satisfy it, require injective maps as a precondition of `pullback`, and reject only statically evident contract violations in normal processing.
10. Add a structured `PhysicalAllocation` result describing physical objects, separate-object key, per-instance element extent or physical domain, alignment, and instance scope, computed per storage-object/instance fiber. Reject nonuniform hardware-instance allocation in the first implementation.
11. Add a runtime-interface record carrying physical dimensions, strides, byte size, alignment, and parameter equations.
12. Make generic `LinearLayout` require an explicit size or physical storage domain.
13. Add target hooks for element-backed vector ABI size/alignment and supported compile-time swizzle forms; image-backed vectors use image-format channel metadata for width.
14. Expose image texel channel count through the image layout or associated storage metadata so factories and code generation can validate image-backed vectors.

### Suggested location

Initially place public layout definitions in `loopy/kernel/array.py` or a new `loopy/kernel/layout.py`. A separate module is preferable once the API stabilizes because `array.py` is already large.

### Tests

- construction, copying, equality, hashing, and mapping for every layout;
- dependency collection;
- static and runtime rejection of every illegal child-type combination, including nested `PrivateLayout`;
- full-environment lowering without positional axis removal;
- totality, range, and named-space alignment checks for piecewise quasi-affine components;
- valid noninjective components whose user-asserted combined map is injective, such as `(floor(i/4), i mod 4)`;
- confirmation that structural validation does not claim to prove injectivity;
- trivial record construction and `make_*_layout` normalization;
- C/F/fixed-stride expression and allocation derivation, including origins and padding;
- generic layout rejection without size;
- element-backed and image-backed vector lowering with piecewise lane expressions;
- scalar-lane singleton and static-swizzle analysis;
- image-vector channel-count validation and rejection beneath scope wrappers;
- separate selector-expression lowering, including separate image vectors;
- image coordinates and local/private/reuse-epoch identity metadata.

### Exit criteria

The shared layout protocol, full named environment, combined-map validation, and closed legal-root union can represent every current core layout feature. Legal compositions do not depend on destructive axis removal, and illegal wrapper combinations remain excluded by child types and factory signatures.

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

1. Build the named projection `PwAff` for that axis and prove that its range is exactly the zero-based compile-time-constant interval required by the legacy tag.
2. Install `make_vector_layout(lane_expr, length, child)` without removing or renumbering any logical dimension.
3. Preserve existing whole-vector behavior for the currently vectorized iname by constructing an explicit static-swizzle proof.
4. Query the target for physical vector size and alignment, including three-vector padding.

### Separate conversion details

For legacy separate axes:

1. Convert each selected axis to a named projection `PwAff` and install `make_separate_layout(selector_exprs, child)`.
2. Initially materialize physical subarguments during preprocessing using the existing naming/ABI strategy.
3. Record the mapping from selector tuples to materialized argument names as lowering output, not canonical array metadata.
4. Restrict the logical shape to each selector fiber and specialize the child under those equalities. Do not remove dimensions positionally; use an explicit named reindexing map only when a dimension is proven redundant after specialization.
5. Require the joint selector range to be a parameter-independent finite constant Cartesian product, enumerate tuples lexicographically, and require selector values to be compile-time known at each scalar access.
6. Permit `make_separate_layout(..., child=make_vector_layout(...))`; exclude the reverse order and repeated wrappers through child types. Shared logical dependencies among selectors and child coordinates are legal when the combined map is injective.

### Tests

- old tag strings produce equivalent layout expressions;
- `tag_array_axes` retains C/F/fixed-stride behavior;
- vector and separate compatibility projections;
- preservation of full named environments during legacy conversion and separate materialization;
- nested separate/vector combinations;
- unsupported compatibility projection diagnostics;
- user-provided generic layout and size.

### Exit criteria

Resolved arrays have one canonical layout. Existing common constructors and `tag_array_axes` still work through conversion.

## Phase 5: Allocation and runtime-wrapper model

### Objectives

Move allocation and host validation from logical shape to layout-provided physical information.

### Tasks

1. Replace `TemporaryVariable.nbytes = product(shape) * itemsize` with layout allocation queries that receive the exact logical shape.
2. Range terminal coordinates per storage-object/instance fiber of the combined layout map. Vector lanes do not multiply vector-object count; separate selectors produce distinct objects; scope keys do not multiply per-instance extent.
3. Reject nonuniform per-hardware-instance allocation requirements in the first implementation.
4. Update C-family temporary declaration sizing.
5. Update base-storage allocation to use:
   - structured per-object element extent;
   - physical storage kind;
   - storage-instance scope without multiplying launch size;

   - alignment and dtype.
6. Implement rectangular runtime interfaces containing:
   - expected physical rank and dimensions;
   - expected strides;
   - allocation size;
   - equations for inferring size parameters.
7. Refactor wrapper parameter inference to consume layout-contributed equations.
8. Preserve existing singleton-axis and empty-array stride equivalence where valid.
9. Preserve rectangular output allocation.
10. For generic layouts:
   - validate byte size/alignment when provided;
   - require explicit runtime interface for output allocation;
   - reject ambiguous parameter inference.
11. Update initializer validation to compare against physical storage representation, not logical shape.

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
- vector physical size and alignment without multiplying by lane count twice;
- lane-correlated child coordinates such as `(floor(i/4), i mod 4)`;
- one allocation per separate selector fiber;
- rejection of nonuniform hardware-instance allocation.

### Exit criteria

No core allocation or wrapper path computes storage as a product of logical shape dimensions.

## Phase 6: Universal address-space transform and minimum callable support

### Objectives

Add `AddressSpace.UNIVERSAL` and an opt-in normalization transform. Do not enable automatic preprocessing conversion until the code-generation, callable, race, and liveness consumers in later phases are ready.

### Tasks

1. Append `UNIVERSAL` without renumbering existing enum values.
2. Replace implicit `max(AddressSpace)` joins with an explicit legacy scope lattice/helper.
3. Implement a two-copy candidate-scope collision query that can test hypothetical private/local/global instance axes without mutating the kernel. Use it for sound temporary-scope inference before universalization.
4. Implement `to_universal_address_space` at translation-unit scope.
5. Convert old global arrays to universal arrays with global linear/image/separate/vector layouts.
6. Convert old local arrays:
   - add canonically named group dimensions to the logical shape;
   - rewrite all accesses through named maps carrying current group coordinates;
   - construct zero-based group-ID projection `PwAff`s and wrap the child layout in `LocalLayout`.
7. Convert old private arrays:
   - add canonically named group and item dimensions;
   - rewrite accesses through named maps carrying current coordinates;
   - construct group/item projection `PwAff`s and wrap in `PrivateLayout`.
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
11. Validate current-instance access constraints and uniform per-instance allocation polyhedrally.
12. Validate named-space alignment and component totality/ranges after universalization; preserve injectivity as a documented provider contract.
13. Make the transform idempotent.
14. Keep automatic preprocessing conversion disabled in this phase; old-address-space code generation remains the default compatibility path.
15. Add an opt-in universal-IR validator for tests and development.

### Preprocessing order

Final target order once automatic conversion is enabled:

1. normalize legacy shapes/layouts;
2. infer `auto` shapes;
3. materialize `SeparateLayout` by selector fiber, expanding callable signatures and call sites consistently when one logical argument spans multiple physical objects;
4. realize ILP/iname-private semantics;
5. infer temporary scope with candidate-scope collision analysis;
6. specialize callable descriptors and callable kernels;
7. universalize all specialized kernels and their call sites;
8. validate universal invariants;
9. run target preprocessing;
10. mark preprocessed.

This ordering is mandatory: universalization adds named instance dimensions and mappings, so independently universalizing unspecialized callers and callees is not supported. Separate materialization must be translation-unit aware rather than deleting selector dimensions independently in each kernel.

### Tests

- global/local/private conversion;
- exact added shape dimensions and constraints;
- named access-map rewriting without dependence on tuple-prefix positions;
- aligned, total, and in-range scope projection expressions;
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

1. Replace `get_access_info` with layout lowering from `LogicalAccess`, preserving named expressions and the exact instruction-domain-to-logical-index map.
2. Compose every layout `PwAff` with the access map and active code-generation domain.
3. Update C-family expression and instruction lowering to dispatch on storage references, lowered coordinate variants, and physical access footprints.
4. Move image coordinate generation into `ImageLayout`; stop reconstructing it from the original index tuple.
5. For scalar accesses, require vector lanes and unmaterialized separate selectors to reduce to compile-time singleton integers.
6. Represent whole-vector access explicitly and prove child storage/coordinate invariance plus a compile-time static swizzle. Reject runtime-dependent or unresolved piecewise lane mappings.
7. Validate that each image-backed vector length equals the image format's channel count before emission.
8. Lower image-vector lane reads as a whole image read followed by static channel selection, and lower whole-vector image accesses as whole-texel operations.
9. Reject every partial image-vector write during code generation. Do not synthesize read-modify-write; only one explicit whole-vector operation with a proved permutation covering all channels exactly once may emit an image store.
10. Consume the storage references selected by early separate materialization.
11. Generate declarations from layout storage kind:
   - global pointer/argument;
   - local/shared declaration;
   - private automatic declaration;
   - image object.
12. Update OpenCL atomic and volatile qualifiers to query physical storage kind and footprint.
13. Update CUDA shared/private/global declaration handling.
14. Update ISPC private-lane duplication to use explicit private layouts.
15. Define `IndexOfCallable` only for layouts with meaningful linear coordinates; reject others clearly.
16. Remove offset application from target code.

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
- scalar lane expressions that become singleton constants only after domain restriction;
- piecewise lane expressions with resolved and unresolved branch cases;
- static identity, reverse, and arbitrary supported read swizzles;
- rejection of runtime-dependent lanes, selectors, and swizzles;
- image-backed `float4` lane reads and whole-vector reads/writes;
- code-generation rejection of scalar-lane and proper-subvector image writes, with no read-modify-write output;
- image-vector channel-count mismatch diagnostics;
- separate arrays, including selector expressions and separate image vectors;
- local/shared and private declarations;
- atomic and volatile accesses;
- unsupported CUDA image diagnostics;
- `indexof` rejection on non-linear layouts.

### Exit criteria

Target expression and declaration code consumes only named-access layout lowering results, does not inspect dim tags or old array address spaces, and never emits runtime vector indexing for a `PwAff` lane.

## Phase 8: Callable descriptors and subarray composition

### Objectives

Preserve polyhedral shape/layout semantics across callable-kernel boundaries.

### Tasks

1. Change resolved `ArrayArgDescriptor` to require a named-set shape and resolved layout. Introduce a separate unresolved specialization descriptor/state where necessary; do not use `shape=None` or `layout=None`.
2. Implement descriptor mapping over shape parameters and layout expressions.
3. Implement descriptor dependency collection from set parameters and layouts.
4. Rewrite `get_arg_descriptor_for_expression`:
   - obtain the exact swept-iname domain;
   - construct the named map into source logical indices;
   - form the exact preimage shape;
   - pull back every layout component, including lane, selector, and scope `PwAff`s;
   - preserve correlated domains and translate parameter namespaces;
   - fix nonswept storage-instance dimensions to current values;
   - require an injective reindexing map as the `pullback` precondition and reject statically evident repeated-element mappings.
5. Replace the minimum Phase 6 callable adapter with full named layout pullback.
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
- composed strided layouts and piecewise vector lanes;
- parameter renaming in layout `PwAff`s;
- repeated-element/noninjective actual rejection;
- scalar selector fixed at a call and swept selectors spanning multiple materialized objects;
- local view passed within current group;
- private view passed within current item;
- scope-changing call rejection;
- vector/separate child descriptors with full named environments;
- nested callable specialization;
- output and in/out arguments;
- symbolic parameter translation;
- third-party `with_descrs` compatibility helper use.

### Exit criteria

No callable descriptor collapses a set shape into independent extents or assumes fixed stride tags.

## Phase 9: Race, ordering, and barrier analysis

### Objectives

Use combined layout maps and physical access footprints for race-related analysis, with universal logical-index equality retained only as a proved-safe scalar optimization.

### Tasks

1. Add a shared builder for:

   ```text
   execution domain
       -> storage object + instance identity + physical access footprint
   ```

2. Canonicalize group/item expressions by hardware tag, including bounds and nonzero bases.
3. Define overlap for scalar elements/lanes, whole vectors, image texels, and target atomic granularity.
4. Replace syntactic write-race checks with a two-copy collision query:
   - duplicate execution coordinates;
   - require a difference in at least one relevant concurrent coordinate;
   - require equal storage object and instance identity;
   - require overlapping schedule-derived live ranges and physical footprints;
   - test nonemptiness.
5. Use equality of universal logical indices only for scalar accesses to the same array under the layout-provider injectivity contract.
6. Reuse and generalize the candidate-scope query introduced in Phase 6; do not maintain a second inference algorithm.
7. Update schedule-time `WriteRaceChecker`, access-range overlap, and variable-access ordering to remove local/global branching and consume footprint relations.
8. Treat unknown/non-affine coordinates or footprints conservatively.
9. For base-storage aliases and callable views with different logical namespaces:
   - compare common physical coordinates and footprints when available;
   - otherwise preserve conservative overlap.
10. Use distinct separate storage-object selectors for disjointness and permit iname-private reuse only after sequentiality and live-range nonoverlap are proved.
11. Remove old `_is_racing_iname_tag` and address-space-specific race logic once unused.

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

Representation-footprint cases:

- distinct scalar lanes of one ordinary vector do not overlap;
- a whole-vector access overlaps every lane;
- an image texel write overlaps every channel read/write;
- distinct materialized separate selectors do not overlap;
- a dynamic or unknown footprint is conservative.

Ordering/barrier cases:

- disjoint injective accesses require no barrier;
- non-affine unknown access is conservative;
- overlapping base-storage aliases conflict;
- provably disjoint base-storage regions avoid unnecessary conflicts if implemented.

### Exit criteria

Race correctness no longer depends on old address-space enum values, tuple positions, mere occurrence of inames in subscripts, or logical-index equality for non-scalar operation footprints.

## Phase 10: Generated-subkernel liveness and storage queries

### Objectives

Replace semantic address-space queries used by scheduling and host allocation, and integrate fine-grained liveness across generated-subkernel boundaries.

### Tasks

1. Replace or redefine:
   - `LoopKernel.global_var_names`;
   - `LoopKernel.local_var_names`;
   - `LoopKernel.local_mem_use`.
2. Replace `_should_temp_var_be_passed` with a liveness query combined with storage kind, instance scope, and ownership.
3. Compute whether each value is live across each generated-subkernel boundary at the finest available program-point granularity.
4. Update generated-subkernel checks:
   - a live `GLOBAL_BUFFER` value may be passed across launches;
   - a live `LOCAL_MEMORY` or `PRIVATE_MEMORY` value cannot cross a launch without explicit save/reload;
   - iname-private reuse requires nonoverlapping epoch live ranges.
5. Update PyOpenCL host allocation and release of global temporaries from their computed live ranges.
6. Update base-storage grouping and nested-base-storage checks.
7. Update save/reload and barrier-related users in core paths.
8. Update local-memory statistics to query layouts.
9. Once Phases 7-10 pass their compatibility suites, enable automatic `to_universal_address_space` in preprocessing and activate the hard pre-codegen nonuniversal-array check.

### Tests

- global temporary across a global barrier;
- local/private temporary rejected across a generated launch boundary;
- base storage across subkernels;
- initialized persistent constants;
- local-memory accounting;
- iname-private reuse accepted for disjoint epoch live ranges and rejected for overlap.

### Exit criteria

Generated-subkernel persistence and allocation/release decisions come from fine-grained liveness plus storage kind/scope rather than `AddressSpace.GLOBAL/LOCAL/PRIVATE` or coarse layout metadata, and automatic preprocessing universalization is enabled without falling back to legacy code generation.

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
   - misaligned, non-total, or out-of-range layout `PwAff`;

   - invalid vector/separate composition;
   - dynamic scalar lane or selector and nonstatic whole-vector swizzle;
   - nonuniform hardware-instance allocation;
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

- named-set logical shapes, `auto` as a construction-time request, and the prohibition on `None` shapes;
- shape parameters and zero-dimensional scalars;
- layouts versus logical shapes;
- the shared `Layout` interface, full named logical-index environment, combined layout map, legal child types, and `make_*_layout` factories;
- linear and rectangular layouts;
- local/private/image layouts;
- piecewise quasi-affine vector lanes and separate selectors;
- static scalar-lane/selector and whole-vector swizzle requirements;
- image texel channel mappings, whole-texel writes, and rejection of partial writes;
- iname-private storage;
- combined-map injectivity and trusted custom layouts;
- universal address space;
- physical storage objects, kinds, instance scopes, ownership, reuse epochs, allocation fibers, and per-instance extent;
- legacy `tag_array_axes` compatibility.

Correct the current Fortran-layout description to column-major.

### `doc/ref_call.rst`

Document:

- polyhedral array descriptors;
- subarray shape/layout pullback through named maps;
- callable reindexing injectivity and specialization;
- current-group/current-item restrictions;
- third-party callable migration helpers.

### `doc/ref_internals.rst`

Document:

- preprocessing order;
- universalization invariants;
- layout-driven lowering and static swizzle proof;
- runtime interface equations and allocation by storage fiber;
- physical access footprints and race maps;
- generated-subkernel liveness, passing, allocation, and release rules.

### Tutorial/examples

Add examples for:

- triangular temporary;
- diamond-shaped tile;
- triangular logical shape in rectangular physical storage;
- packed user-sized linear layout;
- local and private universalized arrays;
- packed vector layout using `(floor(i/w), i mod w)`;
- piecewise static vector swizzles;
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
4. Full named layout environment, combined-map protocol/injectivity, legal-root types, factories, allocation records, and linear/rectangular layouts.
5. Piecewise-affine vector/separate components, static swizzle analysis, allocation behavior, and legacy conversion.
6. Runtime rectangular interface and allocation migration.
7. Candidate-scope race query plus opt-in universalization and minimum callable rewriting.
8. Layout-driven code generation with dual-path compatibility.
9. Full callable descriptor composition.
10. Universal race/order/barrier analysis.
11. Generated-subkernel liveness/storage-query migration and activation of automatic universalization.
12. Cleanup, deprecations, documentation, and examples.

Each pull request should preserve a runnable tree and add its own compatibility tests. Avoid a single flag day that changes shapes, layouts, code generation, and races simultaneously.

## Completion criteria

The project is complete when:

- `None` is never an array shape or resolved callable-descriptor shape, and resolved `.shape` values are named sets;
- triangular, diamond, and union-shaped arrays pass bounds checking;
- all arrays are universal before code generation;
- local/private instances are explicit logical dimensions;
- noncurrent local/private/iname-private accesses are rejected;
- layouts, not dim tags, lower accesses and determine allocation;
- all layout nodes receive one full named logical environment and contribute to one combined map;
- layout components are structurally valid and range-correct, while combined injectivity remains an explicit provider contract;
- illegal child combinations are excluded by types and factories without assigning exclusive ownership of logical axes;
- vector lanes and separate selectors are piecewise quasi-affine expressions with static scalar and whole-vector lowering;
- image texel channels may be represented by an image-backed vector layout without permitting scoped images;
- partial image-vector writes are rejected during code generation;
- allocation is computed by storage-object/instance fiber;
- race checks compare physical operation footprints;
- generic layouts require explicit allocation size;
- rectangular wrappers retain physical shape/stride checks and parameter inference;
- callable descriptors preserve polyhedral domains and composed layouts;
- scalar race checks may optimize through universal logical indices, while vector/image/alias races use physical footprints;
- generated-subkernel passing and allocation decisions use fine-grained liveness plus layout storage kind/scope;
- legacy tuple shapes and `tag_array_axes` continue to work through normalization;
- documentation and persistence formats cover the new model.
