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

### Scope

The execution granularity that owns a distinct storage instance. Scope is not Python/C lexical scope. The initial scopes are global (one instance shared by all workgroups and work items), workgroup (one instance per workgroup), and work item (one instance per work item). Sequential iname-private values do not introduce another physical scope; they introduce reuse epochs that may share one instance when their live ranges do not overlap.

### Physical storage object

One separately named declaration or allocation, such as a kernel argument, temporary buffer, local-memory declaration, private automatic variable, or image object. A `SeparateLayout` may turn one logical array into several physical storage objects.

### Storage instance

One execution-context realization of a physical storage object. A global buffer object has one global instance, a local/shared declaration has one instance per workgroup, and a private declaration has one instance per work item. The **instance key** is the tuple identifying that realization: empty for global scope, group IDs for workgroup scope, and group plus local/item IDs for work-item scope. “Per-instance extent” means the amount of storage in each realization, not the total multiplied by the number of workgroups or work items.

### Instance scope

The `InstanceScope` value identifying which execution coordinates distinguish storage instances. It is part of allocation and alias semantics:

```python
class InstanceScope(Enum):
    GLOBAL = auto()
    WORKGROUP = auto()
    WORK_ITEM = auto()
```

### Storage kind

`StorageKind` identifies the target storage mechanism and declaration/access rules, independently of the logical shape:

```python
class StorageKind(Enum):
    GLOBAL_BUFFER = auto()
    CONSTANT_BUFFER = auto()
    LOCAL_MEMORY = auto()
    PRIVATE_MEMORY = auto()
    IMAGE = auto()
```

- `GLOBAL_BUFFER` is pointer/buffer storage visible to generated device programs;
- `CONSTANT_BUFFER` is read-only storage in a target-distinguished constant
  address space, covering today's `ConstantArg` (`__constant` in OpenCL,
  `__constant__` in CUDA) and initialized read-only temporaries. It is a
  distinct storage kind rather than a form of ownership: it has its own pointer
  qualifier, hardware path, and capacity limits;
- `LOCAL_MEMORY` is workgroup-local/shared storage;
- `PRIVATE_MEMORY` is work-item-local automatic/register-backed storage;
- `IMAGE` is an opaque image object accessed through image operations.

The legal combinations of storage kind and instance scope are constrained by layout types and target support. For example, `LOCAL_MEMORY` has workgroup scope and `PRIVATE_MEMORY` has work-item scope in the first implementation.

### Storage ownership

Ownership says who creates and releases the physical storage object. Array arguments are normally externally owned; Loopy temporaries are normally Loopy-owned; target-defined constants may have static target ownership. Ownership comes from the array declaration or target integration, not from the layout. Together with liveness, it determines whether Loopy allocates, passes, or releases an object.

### Reuse epoch

A reuse epoch distinguishes logically different sequential iname-private values that may share one physical instance. Its **epoch key** participates in logical identity but is deliberately omitted from physical allocation identity. Reuse is legal only when schedule-aware liveness analysis establishes that the epochs' live ranges do not overlap. Generated-subkernel persistence and allocation/release points are derived from liveness, storage kind, instance scope, and ownership rather than stored as coarse layout metadata.

### Physical extent

The amount and arrangement of coordinates available within one storage instance. For linear/vector storage this is commonly an element count plus alignment; for multidimensional or opaque storage it may be a physical storage domain or target-specific dimensions. It is not the number of instances and is not generally derivable from the cardinality or bounding box of the logical shape.

## Logical shape

### Canonical type

The construction-time and resolved types are:

```python
ArrayShape = namedisl.Set | type[auto]
ResolvedArrayShape = namedisl.Set
```

`ArrayBase.shape` may be `auto` only until the exact inference pass runs; every resolved array has a `namedisl.Set`. `None` is never a shape value. Legacy public entry points may accept tuple or string shapes, but must pass them immediately to a shape-construction function that returns `ArrayShape`. Resolved internal shapes must not be represented by a `tuple | namedisl.Set` union. Keep normalization out of object constructors except for a thin backward-compatibility delegation that cannot yet be removed.

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

A legacy tuple entry of `None` is an error; unconstrained dimensions must be expressed explicitly using a named universe set. A top-level legacy `shape=None` is not stored: entry points that explicitly support exact shape inference may translate it immediately to `auto`, and all other entry points reject it with a diagnostic requesting `auto` or an explicit named set. `dim_names` supplies dimension names during conversion and is otherwise subsumed by the named set.

### Shape conventions

- Set dimensions are logical array axes.
- Every logical axis has a unique non-null canonical name. An unnamed axis at zero-based position `i` receives the name `_lpy_s{i}`. `_lpy_` is a protected namespace, so no collision avoidance is needed. A new helper, `get_default_shape_axis_name(i)`, owns this spelling and lives alongside `get_access_map_storage_names`; the latter uses it rather than duplicating the format string.
- Set parameters are integer size parameters.
- Externally visible shape parameters must correspond to integral, read-only `ValueArg`s.
- A zero-dimensional shape is a point `{ [] }`, representing a scalar array consistently with NumPy.
- An empty set represents an array with no valid elements and is distinct from a scalar.
- `auto` is a construction-time request for inference from the exact polyhedral union of accesses. Failure to complete exact inference is an error.
- A rank-known but unchecked array uses an explicitly named universe set such as `{ [i, j] }`.
- Unknown rank is not represented in canonical array IR. A parser or compatibility layer must resolve it before constructing an array or reject the input.

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
array.num_axes                 # attribute; raises for an unresolved `auto` shape
array.rectangular_shape()      # zero-based separable box, or error
array.axis_names
```

`rectangular_shape()` is the compatibility path for code that genuinely requires a NumPy-style shape tuple. It must not silently return a bounding box for a nonrectangular set. No bounding-box shape query is provided.

`auto` has no rank, so `num_axes` and `axis_names` are defined only on a resolved shape and must raise an actionable error on an `auto` array rather than guessing.

Shape stringification uses NumPy shape notation only for a **zero-based** box, for example `shape=(n, m)`, `shape=(n,)`, or `shape=()`. `namedisl.Set.is_box` is true for boxes with nonzero origins as well, so `is_box` alone is not the right predicate: printing `[n] -> { [i] : 2 <= i < n }` as `shape=(n-2,)` would name a shape that `rectangular_shape()` rejects. Non-box shapes and boxes with a nonzero origin are displayed as named sets. Reproducer and persistence formats retain the exact set, including nonzero origins, even when concise display uses extent notation.

### Shape equality and hashing

Set equality must be semantic after named-space alignment, not object identity or textual equality. Parameter order must not affect equality. Logical dimension names are part of the interface and must be treated consistently during equality, hashing, copying, and callable parameter translation.

Persistent hashes should use a stable normalized representation. Equality must remain independent of kernel assumptions so that descriptors remain context-independent values.

## Shape inference and bounds checking

Current inference already computes access ranges and then reduces them to independent minima and maxima. The new representation should retain the exact access range.

For an `auto` temporary:

```text
shape = union of all relevant logical access ranges
```

Every relevant access must be represented successfully; Loopy must not silently omit an unanalyzable access from the inferred shape. An `auto` temporary with no accesses must be removed by an earlier dead-code path or diagnosed; it cannot survive as an unresolved array. Initializer-backed storage contributes its declared physical interface separately.

If access ranges cannot be represented quasi-affinely, inference must fail with an actionable diagnostic. Bounding-box shape inference is not provided: it changes the logical validity set and has no valid shape-inference use case.

Bounds checking becomes:

```text
instruction access range <= array shape
```

Both sets must be aligned by parameter and logical-axis name. Containment is checked under the instruction domain and kernel assumptions. If required parameter alignment or containment cannot be established, the check fails conservatively with a diagnostic. Points inside the rectangular hull but outside a triangular, diamond, or disconnected shape are out of bounds.

## Layout model

### Requirements

All layout nodes obey one shared `Layout` protocol and are evaluated on the same full, named logical-index space. Wrappers never remove or renumber logical axes. A code-generation access carries both named Pymbolic expressions and, when required by a quasi-affine layout component, an exact map from the current instruction domain to the array's logical shape:

```python
@dataclass(frozen=True)
class LogicalAccess:
    """A source-level array access in its named logical-index space.

    :arg domain: The exact active execution domain, including applicable
        instruction predicates and kernel assumptions. It supplies the
        constraints needed to interpret piecewise expressions and prove that
        a lane, selector, or swizzle is static at this access.
    :arg index_exprs: Pymbolic index expressions keyed by logical-axis name.
    """

    domain: namedisl.Set
    index_exprs: constantdict[str, ArithmeticExpression]

    @cached_property
    def index_map(self) -> namedisl.Map | None:
        """Return the exact quasi-affine map from *domain* to logical indices.

        Return *None* if an index expression is not quasi-affine. Layouts that
        contain a ``PwAff`` component reject such an access when they need the
        map. The cached value is derived state and is excluded from equality,
        hashing, and persistence.
        """
        ...

class Layout(Protocol):
    """Shared behavior of immutable logical-to-physical layout values."""

    def map_expr(self, mapper: ExpressionMapper) -> Self:
        """Map the layout's Pymbolic expressions."""
        ...

    def map_parameters(self, mapper: ExpressionMapper) -> Self:
        """Map parameters in Pymbolic and named-ISL components."""
        ...

    def depends_on(self) -> frozenset[str]:
        """Return names on which the layout representation depends."""
        ...

    def update_persistent_hash(self, key_hash, key_builder) -> None:
        """Add the layout's semantic state to a persistent hash."""
        ...

    def align_to_shape(self, logical_shape: namedisl.Set) -> Self:
        """Return this layout aligned to *logical_shape* by name."""
        ...

    def pullback(self, index_map: namedisl.Map) -> Self:
        """Reexpress this layout for a view, subarray, or callable argument.

        If this layout denotes ``L: X -> P`` and *index_map* denotes the named
        reindexing ``f: Y -> X``, return the layout ``L ∘ f: Y -> P``. The
        operation composes every logical-index-dependent component, including
        lane, selector, scope, reuse-epoch, and terminal-coordinate expressions.
        The provider must ensure that ``f`` is injective on the view's shape;
        this method does not prove that precondition.

        Layout components live in two languages: quasi-affine components are
        ``namedisl.PwAff``\\ s, composed with *index_map* directly, while
        terminal expressions are general Pymbolic expressions, for which
        composition is substitution. *index_map* must therefore be a
        single-valued map that is also convertible to per-axis Pymbolic
        expressions; a map that is not (for instance, one with unresolved
        existentially quantified variables) is rejected.
        """
        ...

    def validate(self, logical_shape: namedisl.Set) -> None:
        """Check structural well-formedness relative to *logical_shape*.

        Check canonical named-space alignment, component totality and declared
        ranges, legal child types, and required allocation metadata. Do not
        attempt to prove layout injectivity; injectivity is a provider contract.
        """
        ...

    def lower_access(
            self, access: LogicalAccess,
            context: LayoutLoweringContext) -> LoweredAccess:
        """Lower a logical access to storage, coordinate, and footprint."""
        ...

    def physical_allocation(
            self, logical_shape: namedisl.Set,
            dtype: LoopyType,
            target: TargetBase) -> PhysicalAllocation:
        """Describe the physical objects required for this layout."""
        ...

    def runtime_interface(
            self, logical_shape: namedisl.Set,
            dtype: LoopyType,
            target: TargetBase) -> RuntimeArrayInterface | None:
        """Describe host argument validation/allocation, if supported."""
        ...
```

`LogicalAccess.domain` is needed even though `index_map` is cached from `index_exprs`: expressions alone do not state which piecewise branches are reachable, which loop values are active, or which predicates and assumptions may be used to establish a static lane or selector. The domain and expressions together determine the exact access map. A layout containing a `namedisl.PwAff` component requires the cached map; failure to construct or align it is an actionable error, not a reason to approximate the access.

`map_parameters` covers substitution in both Pymbolic expressions and named-ISL objects. `align_to_shape` canonicalizes a component's input space so that its set dimensions have exactly the logical shape's axis names, with no missing, extra, duplicate, or unnamed dimensions, and its parameter dimensions refer to the same names independent of ordering. This is what “named-space alignment” means here.

Concrete layouts are immutable and hashable values. `validate` is a structural check only. It verifies that components are already in the canonical named space, are defined throughout the exact logical shape, obey declared ranges such as `0 <= lane_expr < length`, use a legal child composition, and provide required allocation metadata. It does **not** prove injectivity. Every layout provider is responsible for satisfying the injectivity contract below; factory construction of a familiar built-in form may make that obligation evident, but validation does not invoke a general collision solver.

A runtime interface describes how a host wrapper recognizes, validates, and, for outputs, allocates a concrete runtime argument: physical rank and dimensions, strides, byte size/alignment, and parameter-inference equations. Returning `None` means that the layout does not define such a host-array ABI. The array may still be usable as an internal temporary, an opaque target object, or an input accepted through custom wrapper code, provided its physical allocation/code-generation requirements are otherwise known. Generic output allocation and standard host argument checks are unavailable without a runtime interface.

The concrete frozen records should have trivial, preferably dataclass-generated constructors. Public `make_*_layout` functions perform compatibility conversion, expression parsing, named-space alignment, normalization, and validation before constructing those records. This keeps policy out of `__init__` and makes construction logic directly testable.

### Combined layout map

Composition is defined by one abstract map over the exact logical shape `S`:

```text
L: S -> (
    storage-object selector,
    storage-instance identity,
    reuse-epoch identity,
    terminal physical coordinate,
    representation coordinate)
```

A terminal contributes the terminal physical coordinate. `SeparateLayout` contributes the storage-object selector. Local/private wrappers contribute storage-instance identity. `InamePrivateLayout` contributes an abstract reuse-epoch key even though physical allocation is reused. `VectorLayout` contributes a vector lane to the representation coordinate. Every node evaluates its component from the unchanged named logical point and passes that same point to its child.

The same logical dimension may contribute to several components. For example, with logical shape `{ [i] : 0 <= i < n }`, a packed vector layout may use `floor(i/4)` as the child's linear coordinate and `i mod 4` as the lane. This is valid because the pair is injective; there is no distinguished axis for the vector wrapper to own or remove.

Built-in quasi-affine components use `namedisl.PwAff` values aligned to `S`. Generic Pymbolic terminal expressions remain available through the trusted custom-layout path when they cannot be represented quasi-affinely. Allocation is computed per storage-object/instance fiber of `L`; the first implementation requires a uniform allocation requirement across hardware instances.

### Compositional type structure

The hierarchy has four semantic levels, listed from the innermost leaf to the outermost wrapper:

1. **Terminal layouts** produce coordinates within one physical storage object. `LinearLayout` and `RectangularLayout` address ordinary element storage; `ImageLayout` produces opaque image coordinates.
2. **Representation layouts** change how values are represented without changing hardware ownership. `VectorLayout` contributes a lane within one vector/texel value, while `SeparateLayout` selects one of several physical storage objects.
3. **Sequential-reuse layout** (`InamePrivateLayout`) distinguishes logical epochs that execute sequentially and may reuse the same physical instance when liveness permits.
4. **Hardware-scope layouts** (`LocalLayout` and `PrivateLayout`) state which workgroup or work item owns a distinct storage instance. They are outermost because storage ownership applies to the complete representation beneath them.

A layout need not contain every level. A plain `LinearLayout` is a complete global layout, while a private vector temporary may have hardware scope outside a vector representation and terminal. The child-type system encodes legal omissions and ordering independently of which logical dimensions the component expressions reference:

```python
ElementTerminalLayout = LinearLayout | RectangularLayout
TerminalLayout = ElementTerminalLayout | ImageLayout

VectorChildLayout = ElementTerminalLayout | ImageLayout
VectorChildT = TypeVar(
    "VectorChildT", bound=VectorChildLayout, covariant=True)
ElementVectorLayout = VectorLayout[ElementTerminalLayout]
ImageVectorLayout = VectorLayout[ImageLayout]
AnyVectorLayout = ElementVectorLayout | ImageVectorLayout

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
InstanceScopedChildLayout = ElementRepresentationLayout | InamePrivateLayout

ArrayLayout = (
    RepresentationLayout
    | InamePrivateLayout
    | LocalLayout
    | PrivateLayout
)
```

`ArrayLayout` is the type accepted by an array; `Layout` is the shared behavioral protocol. A `PrivateLayout` cannot contain another `PrivateLayout` or a `LocalLayout`; hardware scopes cannot nest; representation wrappers cannot contain scopes; `VectorLayout(SeparateLayout(...))` and repeated vector/separate wrappers are excluded; and neither images nor image-backed vectors can appear under a scope wrapper. An `ImageVectorLayout` remains a legal root representation and may be the child of a top-level `SeparateLayout`.

Public construction follows the same state transitions. Factories validate expression spaces and value-level invariants, but do not claim exclusive ownership of logical axes:

```python
make_vector_layout(
    lane_expr: namedisl.PwAff,
    child: VectorChildT, ...) -> VectorLayout[VectorChildT]
make_separate_layout(
    selector_exprs: tuple[namedisl.PwAff, ...],
    child: SeparateChildT, ...) -> SeparateLayout[SeparateChildT]
make_iname_private_layout(
    child: ElementRepresentationLayout, ...) -> InamePrivateLayout
make_local_layout(
    child: InstanceScopedChildLayout, ...) -> LocalLayout
make_private_layout(
    child: InstanceScopedChildLayout, ...) -> PrivateLayout
```

Factories do not accept flattened terminal fields: callers compose `make_local_layout(..., child=make_linear_layout(...))` explicitly. Legacy APIs may offer separate compatibility conversion functions that construct projection `PwAff`s, but canonical records and their constructors remain simple.

### Lowered storage references, coordinates, and footprints

Layout lowering produces one storage reference, a discriminated physical coordinate, and an operation footprint. Nested wrappers must not introduce competing storage names:

```python
@dataclass(frozen=True)
class StorageReference:
    name: str
    kind: StorageKind
    instance_key: tuple[ArithmeticExpression, ...]
    epoch_key: tuple[ArithmeticExpression, ...]

class LoweredCoordinate: ...

@dataclass(frozen=True)
class LinearCoordinate(LoweredCoordinate):
    element_index: ArithmeticExpression

class VectorSelection: ...

@dataclass(frozen=True)
class ScalarLane(VectorSelection):
    lane: int

@dataclass(frozen=True)
class StaticSwizzle(VectorSelection):
    lanes: tuple[int, ...]

@dataclass(frozen=True)
class VectorCoordinate(LoweredCoordinate):
    child: LoweredCoordinate
    selection: VectorSelection

@dataclass(frozen=True)
class ImageCoordinate(LoweredCoordinate):
    coordinates: tuple[ArithmeticExpression, ...]

class AccessFootprint: ...
# Includes scalar element/lane, whole vector, and whole image texel variants.

@dataclass(frozen=True)
class LoweredAccess:
    storage: StorageReference
    coordinate: LoweredCoordinate
    footprint: AccessFootprint
```

A scalar vector access must lower to `ScalarLane`; whole-vector lowering must produce a `StaticSwizzle`. Neither variant carries a runtime lane expression. `SeparateLayout` changes the single `StorageReference` selected by its child. Scope wrappers add instance or reuse-epoch keys. Code generation dispatches on the storage reference, coordinate, and footprint, not on array subclasses or axis tags.

## Terminal layouts

### Linear layout

```python
@dataclass(frozen=True)
class LinearLayout(Layout):
    expr: ArithmeticExpression
    size: ArithmeticExpression | PhysicalStorageDomain
```

`expr` returns a physical element index and is evaluated in the full named logical-index environment. It may share dependencies with wrapper components; only the combined layout map must be injective. `size` is required for generic layouts. An optional base adjustment may be part of the terminal representation, but the legacy `offset` attribute is not retained as independent canonical state.

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

The explicit origin defines how nonzero or negative logical bounds map into physical storage. Each rectangular axis reads its named logical dimension from the full environment; this is a coordinate dependency, not exclusive ownership of that dimension. Legacy `base_indices` translate to origins when they are accepted at the compatibility boundary.

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

Each expression produces one image coordinate from the full named logical-index environment. `physical_shape`, when provided, supports allocation and wrapper validation. Image format, channel type, access mode, texel channel count, and target ABI requirements may be separate storage metadata or fields of the image layout; they must not be inferred from `AddressSpace.UNIVERSAL`. The metadata visible to layout validation and code generation must expose the texel channel count whenever an `ImageLayout` is the child of a `VectorLayout`.

## Scope wrappers

### Hardware-axis mappings

Mappings should identify both the hardware axis and the logical shape dimension:

```python
@dataclass(frozen=True)
class HardwareAxisMapping:
    hardware_axis: int
    logical_expr: namedisl.PwAff
```

`logical_expr` maps the full logical point to the canonical zero-based hardware ID. A named logical axis or positional compatibility input is shorthand for the corresponding projection `PwAff`. Tagged inames with nonzero bases must be normalized explicitly rather than being equated directly with the zero-based hardware ID.

### Local layout

```python
@dataclass(frozen=True)
class LocalLayout(Layout):
    group_mappings: tuple[HardwareAxisMapping, ...]
    child: InstanceScopedChildLayout
```

The complete storage identity is:

```text
(group IDs, child physical coordinate)
```

Each workgroup owns a distinct physical instance. `LocalLayout` contributes the mapped group expressions to the storage-instance key and passes the unchanged logical environment to `child`. Every access must prove that these expressions equal the current workgroup IDs. The child allocation is the per-workgroup allocation and is not multiplied by the number of workgroups. In the first implementation, child physical coordinates, selectors, lanes, and allocation requirements must be independent of the group-instance expressions; this is an explicit uniform-allocation restriction, not axis removal.

Every access must be provably to the current workgroup. A noncurrent or unprovable group access is an error.

### Private layout

```python
@dataclass(frozen=True)
class PrivateLayout(Layout):
    group_mappings: tuple[HardwareAxisMapping, ...]
    local_mappings: tuple[HardwareAxisMapping, ...]
    child: InstanceScopedChildLayout
```

The complete storage identity is:

```text
(group IDs, local/item IDs, child physical coordinate)
```

`PrivateLayout` contributes its mapped group and item expressions to the storage-instance key, proves that they identify the current work item, and passes the unchanged logical environment to `child`. The child allocation is per work item and is not multiplied by launch size. Child physical coordinates, selectors, lanes, and allocation requirements must initially be independent of these instance expressions.

Every access must be provably to the current work item. A noncurrent or unprovable access is an error.

### Iname-private layout

```python
@dataclass(frozen=True)
class InamePrivateMapping:
    iname: str
    logical_expr: namedisl.PwAff

@dataclass(frozen=True)
class InamePrivateLayout(Layout):
    mappings: tuple[InamePrivateMapping, ...]
    child: ElementRepresentationLayout
```

Each mapped sequential iname iteration has a logically distinct value, but the child storage may be reused across iterations. `InamePrivateLayout` contributes the mapped expressions to an abstract reuse-epoch key, proves that an access uses the current iname values, passes the unchanged logical environment to `child`, and reports one child allocation rather than one allocation per epoch; schedule-aware liveness must confirm that the reuse is legal. Child physical coordinates, selectors, lanes, and allocation requirements must initially be independent of the epoch expressions.

The child types admit only the canonical order: an optional hardware instance scope outside optional `InamePrivateLayout`, followed by representation wrappers and a terminal layout. Multiple hardware scope wrappers and scope wrappers inside `VectorLayout` or `SeparateLayout` are not members of `ArrayLayout`; public factories reject dynamically typed attempts to create them.

Mapped inames must be necessarily sequential under the finalized schedule. If this cannot be established, the layout is invalid.

## Refined injectivity contract

Plain physical-address injectivity is incompatible with local/private allocation instances and iname-private storage reuse. The contract applies to the complete combined map `L` on the exact logical shape, after named-space alignment:

> The tuple of storage-object selector, storage-instance key, reuse-epoch key, terminal coordinate, and representation coordinate uniquely identifies a logical point.

Equivalently:

```text
L(x) = L(y)  implies  x = y
```

The epoch key makes the abstract map injective across sequential iname-private epochs even though allocation deliberately drops that key and reuses physical storage.

The property the analyses actually consume is the *physical* one, which does not follow from the abstract contract alone. It is the abstract contract **plus** the schedule-aware liveness proof that justifies dropping the epoch key:

> Restricted to logical points that are concurrently live, the tuple of storage-object selector, storage-instance key, terminal coordinate, and representation coordinate uniquely identifies a logical point.

The representation coordinate must be part of that tuple: two distinct lanes of one vector value share a storage object, instance key, and terminal coordinate, and are distinguished only by the lane.

Injectivity is a semantic contract on every layout provider, not something `validate` attempts to prove. This remains true when all components are quasi-affine: Loopy checks that component expressions are well-formed, total, and in range, but it does not run a general two-copy collision query. A component may be noninjective by itself—`floor(i/4)` and `i mod 4` are each noninjective while their pair is injective—so local checks on individual fields would not establish the contract anyway. Built-in factory documentation states why its standard constructions satisfy the contract; users supplying custom expressions are responsible for the combined map. An optional diagnostic injectivity checker may be added later without becoming part of normal validation.

Pullback through a subarray/reindexing map preserves the contract only when that map is injective on the new logical shape. `pullback` therefore has injective reindexing as a precondition; Loopy rejects mappings that are statically known to repeat elements but does not promise a general proof. Specializing a separate selector preserves injectivity on that selector fiber. Dropping an epoch key is valid only for allocation reuse after schedule-aware liveness proves that the corresponding live ranges cannot overlap.

## Vector layout

Vector storage is a representation wrapper, not a logical shape-axis tag:

```python
@dataclass(frozen=True)
class VectorLayout(Layout, Generic[VectorChildT]):
    lane_expr: namedisl.PwAff
    length: int
    child: VectorChildT
```

`lane_expr` is aligned to the full logical shape and contributes the representation coordinate without changing the environment passed to `child`. It must be total and satisfy `0 <= lane_expr < length` on the exact shape; `length` is a positive compile-time constant. The pair of child outputs and lane is covered by the provider's combined injectivity contract.

For a scalar access, compose `lane_expr` with the instruction-to-logical-index map and restrict it by the active code-generation domain. The result must be provably one compile-time integer, producing `ScalarLane`; dependence on a runtime parameter, unresolved piecewise branch, or nonconstant loop value is rejected. Whole-vector access is a distinct lowering mode. It must prove that storage object, instance key, and child coordinate are invariant across the vectorized instances and that `lane_expr` produces a compile-time-known lane tuple. A whole-vector write requires that tuple to be a permutation of `0..length-1`; reads may use a statically supported swizzle. No runtime vector indexing is introduced by this design.

For an element-terminal child, allocation must account for target vector ABI padding. For example, an OpenCL three-vector may occupy four scalar slots. The layout reports logical vector length, while a target hook reports physical vector storage size and alignment. Generic allocation-size queries may depend on dtype and target for these vector layouts.

For an `ImageLayout` child, `lane_expr` maps the logical point to the vector channels of one image texel. `length` must equal the image format's channel count, such as four for `float4`; this is checked by the factory when the format is known and otherwise before code generation. Ordinary vector ABI padding is not applied to image channels. Lowering a lane read produces `VectorCoordinate(child=ImageCoordinate(...), selection=ScalarLane(...))`, so code generation performs one image read and selects the requested channel. Whole-vector image reads and writes operate on the complete texel value. A write through an image-backed `VectorLayout` is rejected during code generation unless code generation can prove that it writes all lanes exactly once as one whole-vector image store; uncertain coverage is rejected conservatively with an actionable diagnostic. Code generation must not synthesize a read-modify-write for a partial image-vector write.

Composition examples:

```python
make_vector_layout(
    lane_expr="{ [i] -> [(i mod 4)] }", length=4,
    child=make_linear_layout(expr="i // 4", size="ceil(n/4)"))
make_local_layout(
    group_mappings=...,
    child=make_vector_layout(lane_expr=..., child=make_linear_layout(...)))
make_separate_layout(
    selector_exprs=(...,),
    child=make_vector_layout(lane_expr=..., child=make_linear_layout(...)))
make_vector_layout(
    lane_expr="{ [x, y, channel] -> [channel] }", length=4,
    child=make_image_layout(...))
```

## Separate layout

Separate storage is a representation wrapper selecting among distinct storage objects:

```python
@dataclass(frozen=True)
class SeparateLayout(Layout, Generic[SeparateChildT]):
    selector_exprs: tuple[namedisl.PwAff, ...]
    child: SeparateChildT
```

Each selector expression is evaluated on the full logical point and contributes one component of the storage-object selector. The unchanged environment is passed to `child`, and the selector tuple plus child outputs is covered by the provider's combined injectivity contract. For an actual scalar access, composing the selectors with the access map must yield one compile-time selector tuple unless a target-specific indirect-object mechanism is introduced later.

The first implementation requires the joint selector range to be a parameter-independent, finite, compile-time-constant Cartesian product. Selector tuples are enumerated lexicographically and use the existing deterministic subargument naming scheme. Named or positional compatibility axes become projection `PwAff`s. Sparse, correlated, or parameter-dependent selector ranges are deferred because they may require per-fiber child specialization and different allocation sizes.

The first implementation uses **early materialization**: preprocessing creates one physical argument per selector tuple, restricts the logical shape to the corresponding selector fiber, specializes the child under those equalities, and replaces `SeparateLayout` with the specialized child layout. It does not remove dimensions positionally. A dimension may be projected out only through an explicit named reindexing map after proving that no specialized child component depends on it. Late lowering remains a possible future extension, not an alternative in the initial implementation.

The first implementation permits `make_separate_layout(..., child=make_vector_layout(...))`, meaning separate arrays whose entries are vectors. Factory signatures and child annotations exclude the reverse order, repeated vector wrappers, repeated separate wrappers, and scope wrappers inside representation wrappers; no exclusivity restriction is placed on which logical dimensions contribute to the selectors, lane, and child coordinates. These restrictions make lowering and ABI generation deterministic while leaving room for later generalization.

## Physical extent and allocation

Logical shape says which logical values exist; it does not by itself say what to allocate. Allocation planning converts a shape and layout into descriptions of physical storage objects. It does not multiply those descriptions by the number of runtime workgroups or work items.

### Allocation result

```python
@dataclass(frozen=True)
class PhysicalAllocation:
    """All physical storage objects required for one logical array."""

    objects: tuple[PhysicalStorageObject, ...]

@dataclass(frozen=True)
class PhysicalStorageObject:
    """The allocation requirements for one separately named object."""

    object_key: tuple[int, ...] | None
    kind: StorageKind
    element_extent: ArithmeticExpression | PhysicalStorageDomain
    alignment: int | None
    instance_scope: InstanceScope
```

`object_key` is the compile-time selector tuple of a materialized `SeparateLayout`; it is `None` for an unseparated array. `kind` and `instance_scope` use the definitions above. `alignment` is a byte alignment, or `None` when the target/default ABI decides it.

`element_extent` describes one storage instance of this object. An arithmetic expression is the number of terminal storage elements in a one-dimensional allocation. A `PhysicalStorageDomain` is an exact set of valid physical coordinate tuples for storage that is not adequately described by one count. Neither form includes the number of workgroups, work items, separate objects, or sequential epochs.

### From a layout to an allocation

For a fixed object key `o` and instance key `k`, the corresponding **allocation fiber** is:

```text
S[o, k] = { x in logical shape : object(x) = o and instance(x) = k }
```

Allocation proceeds conceptually as follows:

1. Enumerate the finite storage-object selector values. An unseparated layout has one value; a materialized `SeparateLayout` has one value per physical object.
2. For each object, range the terminal physical coordinate over one allocation fiber to obtain the required per-instance extent or physical domain.
3. Check that this requirement is uniform across runtime instance keys. The first implementation rejects a layout whose local/private extent changes by workgroup or work item.
4. Apply representation ABI rules. An element-backed vector changes the terminal element type, size, and alignment; its lanes do not create additional vector objects. Image channels use image-format dimensions rather than ordinary vector padding.
5. Record the storage kind and instance scope. Runtime execution supplies one instance at global, workgroup, or work-item scope as appropriate; the allocation descriptor itself is not replicated.

Examples:

- A global linear temporary yields one `GLOBAL_BUFFER` object with `GLOBAL` instance scope.
- A local temporary yields one `LOCAL_MEMORY` object description; each workgroup receives an instance with the stated per-instance extent.
- A private temporary similarly yields one `PRIVATE_MEMORY` description and one instance per work item.
- A separate layout yields several object descriptions distinguished by `object_key`.
- `InamePrivateLayout` adds no object and no physical instance. Its sequential epochs reuse the enclosing instance.

Every terminal layout must provide one of:

- an explicit element-extent expression;
- an explicit `PhysicalStorageDomain`; or
- structured information from which its factory derives the requirement exactly.

For a first-release rectangular layout with nonnegative strides, the required linear interval runs from zero through the largest declared physical-axis address plus one, including `base_offset`; a negative minimum is rejected. A generic symbolic/strided layout uses its explicit allocation contract instead of asking Loopy to infer an extent. An image layout supplies physical image dimensions whenever Loopy owns the allocation.

`TemporaryVariable.storage_shape`, `base_indices`, and `offset` need not remain compatible public state:

- lower bounds belong in the logical shape;
- offsets belong in the terminal layout expression;
- explicit storage sizing belongs in the layout.

When several arrays share base storage, compatibility includes object key, storage kind, instance scope, alignment, dtype, and physical-coordinate requirements—not merely the largest element count.

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

An unwrapped terminal or representation layout has `GLOBAL_BUFFER` storage kind by default. Array arguments and temporaries differ in ownership and liveness, not in logical address space. `InamePrivateLayout` does not by itself imply hardware-private storage; it proposes sequential reuse of the enclosing global, local, or private instance. Local/private wrappers around images and multiple hardware scope wrappers are rejected initially. Universalization creates named scope dimensions and projection `PwAff`s for their instance mappings; later layout lowering does not depend on those dimensions occupying a tuple prefix.

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

Layouts may expose a runtime interface such as:

```python
@dataclass(frozen=True)
class RuntimeArrayInterface:
    physical_shape: tuple[ArithmeticExpression, ...] | None
    strides: tuple[ArithmeticExpression, ...] | None
    byte_size: ArithmeticExpression | None
    alignment: int | None
```

An array “has no runtime interface” when its resolved layout's `runtime_interface(logical_shape, dtype, target)` returns `None`. Such a layout has no standard host-array contract. This does not make the layout or array invalid: device-only temporaries, opaque target objects, and custom execution wrappers may not need one. It means the standard runtime wrapper cannot infer parameters from that argument, validate its shape/strides beyond separately supplied byte-size information, or allocate it as an output. Those operations require a `RuntimeArrayInterface` or target-specific wrapper logic.

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

Replace `get_access_info` with layout-driven lowering. The access is represented by named expressions plus an instruction-domain-to-logical-index map. Every quasi-affine layout component is composed with that map and restricted by the active code-generation domain. Linear, vector, image, and separate accesses are explicit lowered variants. Offsets and target-axis accumulation are not independently reapplied by code generators.

Scalar vector lanes and separate selectors must reduce to compile-time singleton values. Whole-vector lowering must prove child-coordinate invariance and produce a compile-time `StaticSwizzle`; a runtime-dependent or unresolved piecewise result is rejected. A `VectorCoordinate` whose child is an `ImageCoordinate` lowers lane reads by reading the texel and selecting a channel; whole-vector accesses lower to whole-texel operations. Code generation rejects partial writes to image-backed vectors instead of emitting a read-modify-write.

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
- Every logical access has the correct named space.
- Every layout `PwAff` is aligned with the canonical logical shape, total on that shape, and within its declared range.

- Every local/private/iname-private access is current-instance legal.
- Every terminal layout has an allocation requirement when Loopy allocates it, and hardware-instance fibers have uniform requirements.
- No unlowered legacy dim tags remain.
- Every scalar vector lane and separate selector is a compile-time singleton.
- Every whole-vector access has a proved child-coordinate-invariant static swizzle.
- Every image-backed vector length agrees with the image format's channel count.
- Every write to an image-backed vector is a whole-vector write; code generation rejects partial writes.
- Separate storage has either been materialized by selector fiber or is supported by the target lowering.

## Race and dependency analysis

After universalization, logical shapes include named storage-instance dimensions, but race analysis does not rely on their tuple positions. Compose each instruction access with the combined layout map to obtain:

```text
execution coordinates
    -> storage object + instance identity + physical access footprint
```

The footprint distinguishes a scalar element or vector lane, a whole vector, an image texel, and target-specific atomic granularity. Two accesses may conflict when their storage object and instance identity agree, their schedule-derived live ranges overlap, and their physical footprints overlap. Thus a whole-vector access overlaps every constituent lane, and an image texel write conflicts with reads or writes to any of its channels even though those channels are distinct logical points.

For scalar accesses to one array under the layout-provider injectivity contract, equality of universal logical indices remains a sound optimization. It is not the universal race criterion. This replaces address-space-specific branching and the current syntactic assumption that mentioning a parallel iname in a subscript proves injectivity. Expressions such as `i % 2` and `i-i` require an actual two-copy collision query.

For different array names sharing base storage or callable views with different logical namespaces, compose layouts into a common physical coordinate and footprint when possible or conservatively assume overlap. Distinct separate selectors prove disjointness only when they select distinct physical objects. Iname-private epoch keys permit reuse only after schedule-aware liveness establishes nonoverlapping live ranges.

## Calls to callable kernels

Array argument descriptors become:

```python
@dataclass(frozen=True)
class ArrayArgDescriptor:
    shape: namedisl.Set
    layout: ArrayLayout
    address_space: AddressSpace
```

After normalization, address space is normally `UNIVERSAL`. Any callable-specialization state that does not yet know a shape or layout uses a separate unresolved descriptor type; it does not encode that state as `shape=None` or `layout=None` in `ArrayArgDescriptor`.

For a `SubArrayRef`:

1. Obtain the exact swept-iname domain.
2. Build a named map from callee-visible logical indices to source logical indices.
3. Define the callee shape as the swept domain, and *check* that it is contained in the exact preimage of the source shape. Do not define it as the intersection of the two: intersecting would silently discard the part of the swept domain that lies outside the source array, turning an out-of-bounds subarray reference into a valid, smaller one. The containment check is exactly the bounds check for the reference.
4. Pull back every source-layout component, including `PwAff` lanes/selectors and scope mappings, through this map.
5. Preserve correlated and union domains and translate parameter namespaces explicitly.
6. Fix nonswept storage-instance dimensions to current group/item/iname values.
7. Require the reindexing map to be injective on the callee shape and reject statically evident repeated-element views; `pullback` does not itself prove this precondition.
8. Return the resulting shape and layout with normalized named spaces.

A local view remains tied to the current group. A private view remains tied to the current item. Calls that imply another storage instance are invalid. No implicit global/local/private conversion occurs at a call boundary.

Existing callable specialization behavior should remain: the concrete caller descriptor specializes the callee. Unresolved polyhedral/legacy dual representations must not reach callable code emission.

Third-party `InKernelCallable.with_descrs` implementations may inspect tuple shapes and dim tags directly. Provide migration helpers such as:

```python
descr.rectangular_shape()
descr.linear_strides()
descr.layout
```

and document the source-level compatibility break.

## Generated subkernels and liveness

For device programs split at global barriers, persistence is derived rather than stored in the layout:

- schedule-aware liveness determines whether a value is live across a generated-subkernel boundary;
- a live `GLOBAL_BUFFER` value may use a host/device allocation passed between generated subkernels;
- `LOCAL_MEMORY` and `PRIVATE_MEMORY` instances physically cannot survive a launch boundary, so a value live across that boundary requires an explicit save/reload transformation or is rejected;
- allocation and release points for global temporaries follow their computed live ranges;
- reuse proposed by `InamePrivateLayout` is legal only when the relevant epoch live ranges are nonoverlapping.

Queries currently based on `AddressSpace.GLOBAL` or `.LOCAL` must instead combine layout storage kind and instance scope with liveness and ownership information. This includes temporary passing, local-memory accounting, base-storage checks, and host-side global temporary allocation.

## Compatibility policy

### Retained inputs

Initially retain:

- legacy top-level `shape=None` only at entry points that immediately translate it to supported exact `auto` inference;
- tuple shape input;
- string shape input;
- `dim_names` input;
- `strides`, `order`, and `dim_tags` input;
- old address spaces;
- `tag_array_axes`.

These are conversion front ends, not canonical stored state. Unsupported `shape=None` uses are rejected, and `None` never reaches `ArrayBase.shape` or an array argument descriptor.

### Compatibility properties

A computed `.dim_tags` compatibility view may be offered for layouts exactly representable by old tags. It must return no misleading answer for arbitrary layouts.

A tuple-valued rectangular shape is available through `rectangular_shape()`, not by silently coercing `.shape`.

Breaking compatibility for `storage_shape`, `base_indices`, and `offset` is acceptable. Legacy constructor values may still be translated when inexpensive.

### `tag_array_axes`

`tag_array_axes` parses legacy syntax and replaces the array layout:

- C/F/nesting tags call the rectangular-layout factories;
- fixed strides call `make_strided_layout`;
- `vec` constructs the selected-axis projection `PwAff` and calls `make_vector_layout`;
- `sep` constructs selector projection `PwAff`s and calls `make_separate_layout`;
- new shorthands may construct local/private/iname-private projection mappings or image coordinates.

Add a direct `set_array_layout` API for new code.

## Remaining prototype decisions

The following details remain implementation choices rather than semantic alternatives:

1. The concrete Python type used for `PhysicalStorageDomain`.
2. The division between target-independent vector metadata and the target ABI hook implementation.
3. The exact target hook used to report supported compile-time swizzle forms.
4. The richness of optional generic runtime interfaces beyond byte size and alignment.
5. Whether `.shape` can change directly or requires one transition release through `.index_set`.

Logical axes are canonically named, all layout nodes retain the full named environment, separate layouts are materialized by selector fiber in the first release, and scope/representation wrapper order is fixed as described above.
