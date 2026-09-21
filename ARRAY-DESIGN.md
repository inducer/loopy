# Polyhedral Arrays and Universal Addressing

## Status and scope

This document proposes a new core representation for arrays in Loopy. It covers:

- polyhedral logical shapes;
- universal addressing;
- compositional memory layouts;
- local, private, image, vector, separate, and iname-private storage;
- bounds and race checking;
- code generation and runtime wrappers;
- calls to callable kernels and generated subkernels;
- compatibility with the current public array interface.

The impact on existing transformations is intentionally out of scope, except where a public compatibility entry point such as `tag_array_axes` must retain its behavior.

The intended steady-state model is:

```text
shape = logical named index set
layout = logical index -> storage-instance identity + physical coordinate
address_space = UNIVERSAL before code generation
```

## Goals

1. Permit nonrectangular, parametric array shapes such as triangles and diamonds.
2. Make local and private allocation instances explicit in the logical array model.
3. Replace stored array-axis implementation tags with first-class layouts.
4. Represent vector and separate storage compositionally.
5. Make bounds, race, allocation, callable, and code-generation logic consume one consistent representation.
6. Preserve common rectangular-array behavior, including runtime shape/stride validation and inference of size parameters where possible.
7. Retain legacy constructors and `tag_array_axes` as compatibility front ends where practical.

## Non-goals

- Noninjective layouts are not supported. In particular, a full logical symmetric matrix in which `(i, j)` and `(j, i)` alias is out of scope. A triangular logical shape storing one canonical half remains supported.
- Automatic proof of user-provided layout injectivity is not required. Layout injectivity is initially a trusted contract. A future checker could use ISL, Z3, or another solver.
- Existing transformations are not redesigned here.
- Remote local/private access through shuffles or communication is not initially supported.
- Arbitrary non-quasi-affine shape constraints are not supported.

## Terminology

### Logical shape

The set of valid logical subscript tuples. It says which values exist, not how much storage is allocated.

### Layout

An immutable description of how a valid logical subscript selects a storage instance and a physical coordinate within it.

### Storage instance

A distinct physical allocation induced by execution context. Examples include one local/shared allocation per workgroup and one private allocation per work item.

### Physical extent

The amount and arrangement of physical storage required by a layout. This is not generally derivable from the cardinality or bounding box of the logical shape.

## Logical shape

### Canonical type

The intended final type is:

```python
ArrayBase.shape: namedisl.Set | type[auto] | None
```

Legacy public entry points may accept tuple or string shapes, but must pass them immediately to a shape-construction function that returns the canonical type. Resolved internal shapes must not be represented by a `tuple | namedisl.Set` union. Keep normalization out of object constructors except for a thin backward-compatibility delegation that cannot yet be removed.

If changing `.shape` proves too disruptive in practice, the fallback is to introduce a canonical `.index_set` and temporarily retain `.shape` as a rectangular compatibility view. This is a fallback, not the preferred design.

### Legacy conversion

A legacy shape:

```python
shape=(n, m)
```

becomes the named set equivalent of:

```text
[n, m] -> { [i0, i1] : 0 <= i0 < n and 0 <= i1 < m }
```

A legacy tuple entry of `None` is an error; unconstrained dimensions must be expressed explicitly using a named universe set. `dim_names` supplies dimension names during conversion and is otherwise subsumed by the named set.

### Shape conventions

- Set dimensions are logical array axes.
- Every logical axis has a unique non-null canonical name. An unnamed axis at zero-based position `i` receives the name `_lpy_s{i}`. `_lpy_` is a protected namespace, so no collision avoidance is needed. A new helper, `get_default_shape_axis_name(i)`, owns this spelling and lives alongside `get_access_map_storage_names`; the latter uses it rather than duplicating the format string.
- Set parameters are integer size parameters.
- Externally visible shape parameters must correspond to integral, read-only `ValueArg`s.
- A zero-dimensional shape is a point `{ [] }`, representing a scalar array consistently with NumPy.
- An empty set represents an array with no valid elements and is distinct from a scalar.
- `auto` requests inference from the exact polyhedral union of accesses where possible.
- `None` represents an unresolved shape or rank during early construction.
- A rank-known but unchecked array should use a universe set such as `{ [i, j] }`, not `None`.

### Examples

Triangular:

```text
[n] -> { [i, j] : 0 <= i <= j < n }
```

Diamond:

```text
[n] -> { [i, j] : -n <= i <= n and -n <= j <= n and -n <= i+j <= n and -n <= i-j <= n }
```

A union of regions remains one `namedisl.Set`, not a collection of alternative shapes.

### Shape queries

Provide explicit queries rather than overloading `.shape` with rectangular behavior:

```python
array.num_axes                 # attribute
array.rectangular_shape()      # zero-based separable box, or error
array.axis_names
```

`rectangular_shape()` is the compatibility path for code that genuinely requires a NumPy-style shape tuple. It must not silently return a bounding box for a nonrectangular set. No bounding-box shape query is provided.

Shape stringification checks `namedisl.Set.is_box`. A box is displayed in the usual NumPy shape notation using its axis extents, for example `shape=(n, m)`, `shape=(n,)`, or `shape=()`; non-box shapes are displayed as named sets. Reproducer and persistence formats retain the exact set, including nonzero origins, even when concise display uses extent notation.

### Shape equality and hashing

Set equality must be semantic after named-space alignment, not object identity or textual equality. Parameter order must not affect equality. Logical dimension names are part of the interface and must be treated consistently during equality, hashing, copying, and callable parameter translation.

Persistent hashes should use a stable normalized representation. Equality must remain independent of kernel assumptions so that descriptors remain context-independent values.

## Shape inference and bounds checking

Current inference already computes access ranges and then reduces them to independent minima and maxima. The new representation should retain the exact access range.

For an `auto` temporary:

```text
shape = union of all relevant logical access ranges
```

Every relevant access must be represented successfully; Loopy must not silently omit an unanalyzable access from the inferred shape. A temporary with no accesses remains unresolved or is removed by a separate dead-code path rather than being assigned an arbitrary shape. Initializer-backed storage contributes its declared physical interface separately.

If access ranges cannot be represented quasi-affinely, inference must fail with an actionable diagnostic. Bounding-box shape inference is not provided: it changes the logical validity set and has no valid shape-inference use case.

Bounds checking becomes:

```text
instruction access range <= array shape
```

Both sets must be aligned by parameter and logical-axis name. Containment is checked under the instruction domain and kernel assumptions. If required parameter alignment or containment cannot be established, the check fails conservatively with a diagnostic. Points inside the rectangular hull but outside a triangular, diamond, or disconnected shape are out of bounds.

## Layout model

### Requirements

All layout nodes obey one shared `Layout` protocol. It provides:

```python
class Layout(Protocol):
    def map_expr(self, mapper: ExpressionMapper) -> Self: ...
    def depends_on(self) -> frozenset[str]: ...
    def update_persistent_hash(self, key_hash, key_builder) -> None: ...
    def validate(self, logical_shape: namedisl.Set) -> None: ...
    def lower_access(
            self, logical_index: tuple[ArithmeticExpression, ...],
            context: LayoutLoweringContext) -> LoweredAccess: ...
    def physical_allocation(
            self, dtype: LoopyType,
            target: TargetBase) -> PhysicalAllocation: ...
    def runtime_interface(
            self, dtype: LoopyType,
            target: TargetBase) -> RuntimeArrayInterface | None: ...
```

Concrete layouts are immutable and hashable values. `validate` checks rank, consumed axes, and the layout's injectivity preconditions. `runtime_interface` returns `None` when the layout has no host-array interface. A generic user-provided layout must include an explicit allocation-size expression or physical storage domain; Loopy must not guess generic allocation sizes from the logical shape.

The concrete frozen records should have trivial, preferably dataclass-generated constructors. Public `make_*_layout` functions perform compatibility conversion, expression parsing, normalization, and validation before constructing those records. This keeps policy out of `__init__` and makes construction logic directly testable.

### Compositional structure

Layouts are compositional, but not every layout may contain every other layout. The type system distinguishes terminal coordinates, representation wrappers, sequential scope, and hardware scope:

```python
ElementTerminalLayout = LinearLayout | RectangularLayout
TerminalLayout = ElementTerminalLayout | ImageLayout

# A vector may represent ordinary vector storage or image texel channels.
VectorChildLayout = ElementTerminalLayout | ImageLayout
VectorChildT = TypeVar(
    "VectorChildT", bound=VectorChildLayout, covariant=True)
ElementVectorLayout = VectorLayout[ElementTerminalLayout]
ImageVectorLayout = VectorLayout[ImageLayout]
AnyVectorLayout = ElementVectorLayout | ImageVectorLayout

# SeparateLayout is generic in this deliberately narrow child type.
SeparateChildLayout = TerminalLayout | AnyVectorLayout
SeparateChildT = TypeVar(
    "SeparateChildT", bound=SeparateChildLayout, covariant=True)
RepresentationLayout = (
    TerminalLayout
    | AnyVectorLayout
    | SeparateLayout[SeparateChildLayout]
)

# Images, image vectors, and separated variants of either cannot be scoped.
ElementSeparateLayout = SeparateLayout[
    ElementTerminalLayout | ElementVectorLayout]
ElementRepresentationLayout = (
    ElementTerminalLayout | ElementVectorLayout | ElementSeparateLayout
)

# InamePrivateLayout.child is ElementRepresentationLayout.
InstanceScopedChildLayout = ElementRepresentationLayout | InamePrivateLayout

# LocalLayout.child and PrivateLayout.child are InstanceScopedChildLayout.
ArrayLayout = (
    RepresentationLayout
    | InamePrivateLayout
    | LocalLayout
    | PrivateLayout
)
```

`ArrayLayout` is the type accepted by an array; `Layout` is the shared behavioral protocol. The narrow child types make nonsensical compositions unrepresentable in typed code: a `PrivateLayout` cannot contain another `PrivateLayout` or a `LocalLayout`; hardware scopes cannot nest; representation wrappers cannot contain scopes; `VectorLayout(SeparateLayout(...))` and repeated vector/separate wrappers are excluded; and neither images nor image-backed vectors can appear under a scope wrapper. An `ImageVectorLayout` is nevertheless a legal root representation and may also be the child of a top-level `SeparateLayout`.

Public construction follows the same state transitions:

```python
make_vector_layout(
    child: VectorChildT, ...) -> VectorLayout[VectorChildT]
make_separate_layout(
    child: SeparateChildT, ...) -> SeparateLayout[SeparateChildT]
make_iname_private_layout(
    child: ElementRepresentationLayout, ...) -> InamePrivateLayout
make_local_layout(
    child: InstanceScopedChildLayout, ...) -> LocalLayout
make_private_layout(
    child: InstanceScopedChildLayout, ...) -> PrivateLayout
```

Factories reject overlapping consumed axes and other value-level violations. They do not accept flattened terminal fields: callers compose `make_local_layout(..., child=make_linear_layout(...))` explicitly. Legacy APIs may offer separate compatibility conversion functions, but canonical records and their constructors remain simple.

### Lowered storage references and accesses

Layout lowering should produce one storage reference plus a discriminated coordinate, rather than the current undifferentiated `AccessInfo`. Nested wrappers must not introduce competing storage names:

```python
@dataclass(frozen=True)
class StorageReference:
    name: str
    kind: StorageKind
    instance_axes: tuple[InstanceAxis, ...]
    lifetime: StorageLifetime

class LoweredCoordinate: ...

@dataclass(frozen=True)
class LinearCoordinate(LoweredCoordinate):
    element_index: ArithmeticExpression

@dataclass(frozen=True)
class VectorCoordinate(LoweredCoordinate):
    child: LoweredCoordinate
    lane: ArithmeticExpression

@dataclass(frozen=True)
class ImageCoordinate(LoweredCoordinate):
    coordinates: tuple[ArithmeticExpression, ...]

@dataclass(frozen=True)
class LoweredAccess:
    storage: StorageReference
    coordinate: LoweredCoordinate
```

`SeparateLayout` changes the single `StorageReference` selected by its child; it does not wrap a second storage name around an already-named access. Scope wrappers update storage kind, instance axes, and lifetime. Code generation dispatches on the storage reference and lowered coordinate, not on array subclasses or axis tags.

## Terminal layouts

### Linear layout

```python
@dataclass(frozen=True)
class LinearLayout(Layout):
    expr: ArithmeticExpression
    size: ArithmeticExpression | PhysicalStorageDomain
```

`expr` returns a physical element index. `size` is required for generic layouts. An optional base adjustment may be part of the terminal representation, but the legacy `offset` attribute is not retained as independent canonical state.

### Rectangular layout

Rectangular layouts are common enough to deserve structured subclasses:

```python
@dataclass(frozen=True)
class RectangularLayout(Layout):
    axes: tuple[RectangularPhysicalAxis, ...]
    base_offset: ArithmeticExpression = 0

@dataclass(frozen=True)
class RectangularPhysicalAxis:
    logical_axis: str
    origin: ArithmeticExpression
    extent: ArithmeticExpression
    stride: ArithmeticExpression
```

`RectangularLayout` is a separate layout with computed `expr` and allocation properties, not a dataclass subclass requiring callers to supply inherited `expr` and `size` fields. Its address expression is:

```text
base_offset + sum((logical_axis - origin) * stride)
```

The explicit origin defines how nonzero or negative logical bounds map into physical storage. Legacy `base_indices` translate to origins when they are accepted at the compatibility boundary.

Construction conveniences are functions:

```python
make_c_layout(axes=...)
make_f_layout(axes=...)
make_strided_layout(axes=..., strides=..., origins=...)
```

The C/F factories use the rectangular physical axes supplied by the caller; they do not attempt to derive axis origins or extents from a correlated logical set. They derive their address expression and physical footprint automatically and retain enough structure for runtime wrappers to:

- validate host array shape and strides;
- infer size parameters from observed dimensions and strides;
- allocate output arrays;
- preserve existing rectangular behavior.

For a nonrectangular logical shape, a rectangular layout describes storage for a rectangular physical container. Holes in the logical set remain unused. C/F order does not imply packed triangular storage.

The first implementation supports nonnegative strides and requires the derived element index to be nonnegative within the declared physical axes. Negative strides require an explicit generic `LinearLayout` with a user-provided allocation and pointer-origin contract. A built-in rectangular layout rejects statically evident overlapping strides; symbolic cases that cannot be established from the standard C/F construction must use the explicitly trusted custom-layout path.

### Image layout

```python
@dataclass(frozen=True)
class ImageLayout(Layout):
    axis_exprs: tuple[ArithmeticExpression, ...]
    physical_shape: tuple[ArithmeticExpression, ...] | None
```

Each expression produces one image coordinate. `physical_shape`, when provided, supports allocation and wrapper validation. Image format, channel type, access mode, texel channel count, and target ABI requirements may be separate storage metadata or fields of the image layout; they must not be inferred from `AddressSpace.UNIVERSAL`. The metadata visible to layout validation and code generation must expose the texel channel count whenever an `ImageLayout` is the child of a `VectorLayout`.

## Scope wrappers

### Hardware-axis mappings

Mappings should identify both the hardware axis and the logical shape dimension:

```python
@dataclass(frozen=True)
class HardwareAxisMapping:
    hardware_axis: int
    array_axis: str | int
```

Named logical axes are preferred; positional indices may be accepted by construction factories as conveniences.

### Local layout

```python
@dataclass(frozen=True)
class LocalLayout(Layout):
    group_axes: tuple[HardwareAxisMapping, ...]
    child: InstanceScopedChildLayout
```

The complete storage identity is:

```text
(group IDs, child physical coordinate)
```

Each workgroup owns a distinct physical instance. `LocalLayout` consumes the mapped group axes: it validates them, removes them from the logical index tuple, and passes only the remaining axes to `child`. The child allocation is the per-workgroup allocation and is not multiplied by the number of workgroups. Initially, the child physical-coordinate expression must not depend on consumed group-instance dimensions.

Every access must be provably to the current workgroup. A noncurrent or unprovable group access is an error.

### Private layout

```python
@dataclass(frozen=True)
class PrivateLayout(Layout):
    group_axes: tuple[HardwareAxisMapping, ...]
    local_axes: tuple[HardwareAxisMapping, ...]
    child: InstanceScopedChildLayout
```

The complete storage identity is:

```text
(group IDs, local/item IDs, child physical coordinate)
```

`PrivateLayout` consumes its mapped group and item axes, validates them against the current work item, removes them, and lowers the remaining tuple through `child`. The child allocation is per work item and is not multiplied by launch size.

Every access must be provably to the current work item. A noncurrent or unprovable access is an error.

### Iname-private layout

```python
@dataclass(frozen=True)
class InamePrivateLayout(Layout):
    axes: tuple[InamePrivateAxis, ...]
    child: ElementRepresentationLayout
```

Each mapped sequential iname iteration has a logically distinct value, but the child storage may be reused across iterations. `InamePrivateLayout` validates and consumes its mapped axes, removes them before lowering through `child`, and reports one child allocation rather than one allocation per iname value. An access must use the current value of each mapped iname.

The child types admit only the canonical order: an optional hardware instance scope outside optional `InamePrivateLayout`, followed by representation wrappers and a terminal layout. Multiple hardware scope wrappers and scope wrappers inside `VectorLayout` or `SeparateLayout` are not members of `ArrayLayout`; public factories reject dynamically typed attempts to create them.

Mapped inames must be necessarily sequential under the finalized schedule. If this cannot be established, the layout is invalid.

## Refined injectivity contract

Plain physical-address injectivity is incompatible with local/private allocation instances and iname-private storage reuse. The contract is instead:

> A layout is injective in logical storage-instance space, including workgroup, work-item, and nonoverlapping sequential-lifetime coordinates.

For concurrently live values:

```text
same storage-instance identity and physical coordinate
    if and only if
same logical array index
```

Physical coordinates may be reused across sequential iname-private instances only because their lifetimes cannot overlap.

User-provided layout injectivity is trusted initially. Built-in layouts must be constructed to satisfy the contract. Noninjective legacy layouts, such as zero stride on a nonsingleton axis or overlapping multidimensional strides, violate the new contract and must not silently enter the normalized IR.

## Vector layout

Vector storage is a representation wrapper, not a logical shape-axis tag:

```python
@dataclass(frozen=True)
class VectorLayout(Layout, Generic[VectorChildT]):
    axis: str | int
    length: int
    child: VectorChildT
```

Semantics:

1. The selected logical axis is removed before lowering through `child`.
2. Its value becomes the vector lane.
3. `child` selects the vector storage object and vector element coordinate.
4. Lowering returns the child's single storage reference with a `VectorCoordinate(child_coordinate, lane)`.

In the first implementation, the selected axis must be exactly the zero-based interval `0 <= lane < length`, with compile-time constant `length`, for every child coordinate on which it is valid. Nonzero-based, noncontiguous, or child-correlated lane domains require a future explicit value-to-lane map. Lane access must be compile-time constant unless it corresponds to the currently vectorized iname and the target supports whole-vector evaluation.

For an element-terminal child, allocation must account for target vector ABI padding. For example, an OpenCL three-vector may occupy four scalar slots. The layout reports logical vector length, while a target hook reports physical vector storage size and alignment. Generic allocation-size queries may depend on dtype and target for these vector layouts.

For an `ImageLayout` child, the selected axis maps to the vector channels of one image texel. `length` must equal the image format's channel count, such as four for `float4`; this is checked by the factory when the format is known and otherwise before code generation. Ordinary vector ABI padding is not applied to image channels. Lowering a lane read produces `VectorCoordinate(child=ImageCoordinate(...), lane=...)`, so code generation performs one image read and selects the requested channel. Whole-vector image reads and writes operate on the complete texel value. A write through an image-backed `VectorLayout` is rejected during code generation unless code generation can prove that it writes all lanes exactly once as one whole-vector image store; uncertain coverage is rejected conservatively with an actionable diagnostic. Code generation must not synthesize a read-modify-write for a partial image-vector write.

Composition examples:

```python
make_vector_layout(axis="lane", child=make_linear_layout(...))
make_local_layout(
    group_axes=...,
    child=make_vector_layout(axis="lane", child=make_linear_layout(...)))
make_separate_layout(
    axes=("field",),
    child=make_vector_layout(axis="lane", child=make_linear_layout(...)))
make_vector_layout(
    axis="channel", length=4, child=make_image_layout(...))
```

## Separate layout

Separate storage is a representation wrapper selecting among distinct storage objects:

```python
@dataclass(frozen=True)
class SeparateLayout(Layout, Generic[SeparateChildT]):
    axes: tuple[str | int, ...]
    child: SeparateChildT
```

Semantics:

1. Values of the selected logical axes choose a physical storage object.
2. Selected axes are removed before lowering through `child`.
3. Lowering returns a storage selection plus the child access.
4. The selector must be compile-time constant at code generation unless a target-specific indirect-object mechanism is introduced later.

The first implementation requires each selected axis to be a parameter-independent, zero-based, compile-time-constant interval, and requires the selector axes to form a Cartesian product independent of the remaining child domain. Selector tuples are enumerated lexicographically and use the existing deterministic subargument naming scheme. Parameter-dependent, sparse, or correlated selector sets are deferred because they may require different child shapes and allocation sizes.

The first implementation uses **early materialization**: preprocessing creates one physical argument per selector tuple, as today, and replaces `SeparateLayout` with ordinary child layouts on those arguments. Late lowering remains a possible future extension, not an alternative in the initial implementation.

The first implementation permits `make_separate_layout(..., child=make_vector_layout(...))`, meaning separate arrays whose entries are vectors. Factory signatures and child annotations exclude the reverse order, repeated vector wrappers, repeated separate wrappers, and scope wrappers inside representation wrappers; factories reject overlapping consumed axes. These restrictions make lowering and ABI generation deterministic while leaving room for later generalization.

## Physical extent and allocation

Logical shape does not determine allocation size. Layout allocation is a structured result, not one undifferentiated scalar:

```python
@dataclass(frozen=True)
class PhysicalAllocation:
    objects: tuple[PhysicalStorageObject, ...]

@dataclass(frozen=True)
class PhysicalStorageObject:
    kind: StorageKind
    element_extent: ArithmeticExpression | PhysicalStorageDomain
    alignment: int | None
    instance_scope: InstanceScope
    lifetime: StorageLifetime
```

`element_extent` is measured in the terminal storage element type; byte size is derived from dtype and, for vector storage, the target ABI. Instance multiplicity is represented by `instance_scope`, not multiplied into per-instance extent. `SeparateLayout` yields one object per materialized selector tuple. Local/private wrappers change scope and lifetime without multiplying the child extent.

Every terminal layout must provide either:

- an explicit element-extent expression;
- an explicit physical storage domain; or
- structured information from which a built-in layout derives the extent exactly.

For first-release rectangular layouts with nonnegative strides, the footprint is the half-open interval from zero through the largest declared physical-axis address plus one, including `base_offset`; a negative minimum is rejected. Layouts returned by the C/F factories are injective by construction. Generic strided layouts outside these rules use the trusted custom-layout path and an explicit allocation contract. Image layouts require physical dimensions when Loopy allocates them. Vector layouts adjust child element type, extent, and alignment according to the target ABI.

`TemporaryVariable.storage_shape`, `base_indices`, and `offset` need not remain compatible public state:

- lower bounds belong in the logical shape;
- offsets belong in the terminal layout expression;
- explicit storage sizing belongs in the layout.

Base-storage allocation must additionally account for storage kind, storage-instance scope, lifetime, alignment, and dtype.

## Universal address space

Add `AddressSpace.UNIVERSAL`, preserving existing numeric values by appending it if `AddressSpace` remains an `IntEnum`.

`UNIVERSAL` means:

> The logical array has been normalized so that storage-instance dimensions and physical representation are described by its shape and layout.

It does not mean OpenCL `__generic`, CUDA global memory, or unqualified C storage.

The old address-space ordering cannot include `UNIVERSAL`. Code using `max(address_space)` as a scope join must be replaced with an explicit old-scope inference operation before universalization.

## Universalization transform

`to_universal_address_space` is a translation-unit transform and must be idempotent.

### Global conversion

```text
old shape: S
old layout: f(i)
new shape: S
new layout: terminal/representation layout f(i)
new address space: UNIVERSAL
```

An unwrapped terminal or representation layout has global/persistent storage kind by default. Array arguments and persistent temporaries differ in ownership and lifetime metadata, not in logical address space. `InamePrivateLayout` does not by itself imply hardware-private storage; it refines the lifetime of the enclosing global, local, or private storage. Local/private wrappers around images and multiple hardware scope wrappers are rejected initially.

No execution-instance dimensions are added.

### Local conversion

```text
old access: A[i...]
new access: A[g0, g1, ..., i...]
new shape: group domain x old shape
new layout: make_local_layout(group mappings, old physical layout)
```

### Private conversion

```text
old access: A[i...]
new access: A[g0, ..., l0, ..., i...]
new shape: group domain x local domain x old shape
new layout: make_private_layout(group mappings, local mappings, old physical layout)
```

Canonical group/item coordinates must match code generation, including hardware inames with nonzero bases.

### Iname-private conversion

Sequential private axes are present in the logical shape and every access uses the current iname value. `InamePrivateLayout` records that the physical allocation may be reused.

### Validation

For every access, Loopy must prove:

- local group coordinates equal current group IDs;
- private group and item coordinates equal the current work item;
- iname-private coordinates equal current sequential iname values.

Failure or inability to prove equality is an error. Future shuffle/communication lowering may relax this.

## Runtime wrappers

Runtime wrappers validate physical storage, not the logical shape directly.

Layouts should expose a runtime interface such as:

```python
@dataclass(frozen=True)
class RuntimeArrayInterface:
    physical_shape: tuple[ArithmeticExpression, ...] | None
    strides: tuple[ArithmeticExpression, ...] | None
    byte_size: ArithmeticExpression | None
    alignment: int | None
```

Rectangular layouts must continue to support:

- validation of runtime rank, dimensions, and strides;
- inference of integral size parameters from runtime arrays;
- output allocation;
- existing singleton- and empty-axis stride rules where applicable;
- `skip_arg_checks` behavior.

Parameter inference should be expressed as equations contributed by the layout. For example, a physical extent `n + 2` contributes an equation against the observed runtime dimension. Existing integer solving can then continue where equations are unambiguous.

For generic layouts:

- the required size permits a one-dimensional internal allocation, but does not by itself define a host-array ABI;
- wrappers may validate byte size and alignment if a runtime interface is supplied;
- shape-parameter inference is available only when the layout provides equations;
- output allocation requires an explicit runtime physical interface;
- ambiguous inference or allocation must produce a targeted error, not a guessed layout.

Logical bounds remain compile-time/polyhedral checks and are separate from runtime physical-storage checks.

## Code generation

Code generation assumes all arrays have `AddressSpace.UNIVERSAL` and resolved layouts.

### Access lowering

Replace `get_access_info` with layout-driven lowering. Linear, vector, image, and separate accesses are explicit lowered variants. Offsets and target-axis accumulation are not independently reapplied by code generators. A `VectorCoordinate` whose child is an `ImageCoordinate` lowers lane reads by reading the texel and selecting a channel; whole-vector accesses lower to whole-texel operations. Code generation rejects partial writes to image-backed vectors instead of emitting a read-modify-write.

### Declarations

Declarations derive from layout/storage kind:

| Layout/storage | OpenCL | CUDA |
|---|---|---|
| Global linear argument | `__global T *` | pointer kernel parameter |
| Persistent global temporary | host/device allocation passed to kernels | device allocation passed to kernels |
| Local layout | `__local` allocation | `__shared__` allocation |
| Private layout | automatic storage | thread-local/register-backed automatic storage |
| Image layout | image object and image intrinsics | unsupported until texture/surface support exists |

Atomics and volatile casts must query lowered physical storage kind, not `address_space`.

### Pre-codegen invariants

- Every array address space is `UNIVERSAL`.
- No `auto` shape, layout, stride, or offset remains.
- Every logical access has the correct rank.
- Every local/private/iname-private access is current-instance legal.
- Every terminal layout has an allocation requirement when Loopy allocates it.
- No unlowered legacy dim tags remain.
- Vector lane information occurs exactly once.
- Every image-backed vector length agrees with the image format's channel count.
- Every write to an image-backed vector is a whole-vector write; code generation rejects partial writes.
- Separate storage has either been materialized or is supported by the target lowering.

## Race and dependency analysis

After universalization, logical indices include storage-instance dimensions:

```text
global:  user index
local:   group IDs + user index
private: group IDs + item IDs + user index
```

Under the refined injectivity contract, equal concurrently live physical locations correspond to equal universal logical indices. Race checks can therefore use one model for all layouts.

Construct a relation:

```text
execution coordinates -> universal logical array indices
```

For two access instances, ask whether distinct relevant concurrent coordinates can produce equal universal logical indices with overlapping lifetimes.

This replaces address-space-specific branching in race analysis. It also replaces the current syntactic assumption that mentioning a parallel iname in a subscript proves injectivity. Expressions such as `i % 2` and `i-i` require an actual collision query.

For different array names sharing base storage, compose layouts into a common physical coordinate when possible or conservatively assume overlap.

Iname-private logical indices may map to reused physical storage because different mapped iname values are necessarily sequential and have nonoverlapping lifetimes.

## Calls to callable kernels

Array argument descriptors become:

```python
@dataclass(frozen=True)
class ArrayArgDescriptor:
    shape: namedisl.Set | None
    layout: ArrayLayout | None
    address_space: AddressSpace
```

After normalization, address space is normally `UNIVERSAL`.

For a `SubArrayRef`:

1. Obtain the exact swept-iname domain.
2. Build a map from callee-visible indices to source logical indices.
3. Compose the source layout with this map.
4. Preserve correlated and union domains.
5. Fix nonswept storage-instance dimensions to current group/item/iname values.
6. Return the resulting shape and layout.

A local view remains tied to the current group. A private view remains tied to the current item. Calls that imply another storage instance are invalid. No implicit global/local/private conversion occurs at a call boundary.

Existing callable specialization behavior should remain: the concrete caller descriptor specializes the callee. Unresolved polyhedral/legacy dual representations must not reach callable code emission.

Third-party `InKernelCallable.with_descrs` implementations may inspect tuple shapes and dim tags directly. Provide migration helpers such as:

```python
descr.rectangular_shape()
descr.linear_strides()
descr.layout
```

and document the source-level compatibility break.

## Generated subkernels and lifetime

For device programs split at global barriers:

- persistent global allocations may be passed between generated subkernels;
- local allocations cannot survive a launch boundary;
- private allocations cannot survive a launch boundary;
- iname-private allocations cannot escape their sequential lifetime.

Queries currently based on `AddressSpace.GLOBAL` or `.LOCAL` must instead use layout storage kind and lifetime. This includes temporary passing, local-memory accounting, base-storage checks, and host-side global temporary allocation.

## Compatibility policy

### Retained inputs

Initially retain:

- tuple shape input;
- string shape input;
- `dim_names` input;
- `strides`, `order`, and `dim_tags` input;
- old address spaces;
- `tag_array_axes`.

These are conversion front ends, not canonical stored state.

### Compatibility properties

A computed `.dim_tags` compatibility view may be offered for layouts exactly representable by old tags. It must return no misleading answer for arbitrary layouts.

A tuple-valued rectangular shape is available through `rectangular_shape()`, not by silently coercing `.shape`.

Breaking compatibility for `storage_shape`, `base_indices`, and `offset` is acceptable. Legacy constructor values may still be translated when inexpensive.

### `tag_array_axes`

`tag_array_axes` parses legacy syntax and replaces the array layout:

- C/F/nesting tags call the rectangular-layout factories;
- fixed strides call `make_strided_layout`;
- `vec` calls `make_vector_layout`;
- `sep` calls `make_separate_layout`;
- new shorthands may mark local/private/iname-private axes or image coordinates.

Add a direct `set_array_layout` API for new code.

## Remaining prototype decisions

The following details remain implementation choices rather than semantic alternatives:

1. The concrete Python type used for `PhysicalStorageDomain`.
2. The division between target-independent vector metadata and the target ABI hook implementation.
3. The richness of optional generic runtime interfaces beyond byte size and alignment.
4. Whether `.shape` can change directly or requires one transition release through `.index_set`.

Logical axes are canonically named, separate layouts are materialized early in the first release, and scope/representation wrapper order is fixed as described above.
