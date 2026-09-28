# Polyhedral Arrays and Explicit Storage

## Status and scope

This document proposes a new core representation for arrays in Loopy. It covers:

- polyhedral logical shapes;
- a new canonical `Array` value, held by (rather than inherited by) arguments and temporaries;
- compositional memory layouts;
- explicit storage instances for workgroup-local, work-item-private, and sequentially reused storage;
- image, vector, and separate storage;
- bounds and race checking;
- code generation and runtime wrappers;
- calls to callable kernels and generated subkernels;
- compatibility with the current public array interface.

The impact on existing transformations is intentionally out of scope, except where a public compatibility entry point such as `tag_array_axes` must retain its behavior. A migration plan for `loopy/transform/` is deferred; see “Deferred: transformation migration” in `ARRAY-IMPL.md`.

The intended steady-state model is:

```text
Array = (shape, layout)
shape  = logical named index set, including explicit storage-instance axes
layout = logical index -> storage-instance identity + physical coordinate
```

An array's *role* — argument versus temporary, input versus output, ownership, initializer, base storage — lives on the object that *holds* the `Array`, not in the `Array` itself.

## Goals

1. Permit nonrectangular, parametric array shapes such as triangles and diamonds.
2. Make storage instances explicit in the logical array model, so that a future `compute` transformation and eventual shuffle/inter-group communication support can address them directly.
3. Replace stored array-axis implementation tags with first-class layouts.
4. Represent vector and separate storage compositionally.
5. Make bounds, race, allocation, callable, and code-generation logic consume one consistent representation.
6. Introduce the new semantics as a new array *type* rather than as a new address space, and use composition rather than inheritance to avoid proliferating array classes.
7. Preserve common rectangular-array behavior, including runtime shape/stride validation and inference of size parameters where possible.
8. Retain legacy constructors and `tag_array_axes` as compatibility front ends where practical.

## Non-goals

- Noninjective layouts are not supported. In particular, a full logical symmetric matrix in which `(i, j)` and `(j, i)` alias is out of scope. A triangular logical shape storing one canonical half remains supported.
- Automatic proof of user-provided layout injectivity is not required. Layout injectivity is a trusted contract. A future checker could use ISL, Z3, or another solver.
- Existing transformations are not redesigned here.
- Remote local/private access through shuffles or communication is not *initially* supported, but the representation is chosen so that it can be added without changing the shape model.
- Arbitrary non-quasi-affine shape constraints are not supported.
- Physical-layout-level race and alias analysis is not attempted. Race analysis reasons at the logical level only; see “Race and dependency analysis”.
- Vector swizzles beyond the identity permutation are not supported initially.
- Image texel channels are not exposed as a logical axis initially.

## Terminology

### Logical shape

The set of valid logical subscript tuples. It says which values exist, not how much storage is allocated. After storage instances are made explicit, it also includes the axes that identify *which* workgroup, work item, or sequential epoch owns a value.

### Layout

An immutable description of how a valid logical subscript selects a storage instance and a physical coordinate within it.

### Physical storage object

One separately named declaration or allocation, such as a kernel argument, temporary buffer, local-memory declaration, private automatic variable, or image object. A `SeparateLayout` may turn one logical array into several physical storage objects.

### Storage instance

One execution-context realization of a physical storage object. A global buffer object has one global instance, a local/shared declaration has one instance per workgroup, and a private declaration has one instance per work item. The **instance key** is the tuple identifying that realization: empty for global scope, group IDs for workgroup scope, and group plus local/item IDs for work-item scope. “Per-instance extent” means the amount of storage in each realization, not the total multiplied by the number of workgroups or work items.

### Instance scope

The `InstanceScope` value identifying which execution coordinates distinguish storage instances:

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
- `CONSTANT_BUFFER` is read-only storage in a target-distinguished constant address space, covering today's `ConstantArg` (`__constant` in OpenCL, `__constant__` in CUDA) and initialized read-only temporaries. It is a distinct storage kind rather than a form of ownership: it has its own pointer qualifier, hardware path, and capacity limits;
- `LOCAL_MEMORY` is workgroup-local/shared storage;
- `PRIVATE_MEMORY` is work-item-local automatic/register-backed storage;
- `IMAGE` is an opaque image object accessed through image operations.

**Both `StorageKind` and `InstanceScope` are derived, not stored.** They are computed from the layout tree: an `ImageLayout` terminal gives `IMAGE`; an `InstancedLayout` with a hardware instance gives `LOCAL_MEMORY`/`PRIVATE_MEMORY` and the corresponding scope; a read-only array with the constant-storage request gives `CONSTANT_BUFFER`; otherwise `GLOBAL_BUFFER` with `GLOBAL` scope. Storing them alongside the layout would create two sources of truth that could disagree. They are exposed as properties on the layout and on the allocation records that code generation consumes.

### Storage ownership

Ownership says who creates and releases the physical storage object. Array arguments are normally externally owned; Loopy temporaries are normally Loopy-owned; target-defined constants may have static target ownership. Ownership comes from the array's *holder* — the argument or temporary — not from the layout. Together with liveness, it determines whether Loopy allocates, passes, or releases an object.

### Reuse epoch

A reuse epoch distinguishes logically different sequential values that may share one physical instance. Its **epoch key** participates in logical identity but is deliberately omitted from physical allocation identity. Reuse is legal only when schedule-aware liveness analysis establishes that the epochs' live ranges do not overlap. Generated-subkernel persistence and allocation/release points are derived from liveness, storage kind, instance scope, and ownership rather than stored as coarse layout metadata.

### Physical extent

The number of terminal storage elements available within one storage instance. It is **always declared or structurally derived, never inferred by ranging an address expression** (see “Physical extent and allocation”). It is not the number of instances.

## The `Array` value

### Motivation

The new semantics are introduced as a new *type*, not as a new address space. An earlier draft added `AddressSpace.UNIVERSAL`; that was rejected for three reasons:

1. `AddressSpace` is an `IntEnum` whose ordering is load-bearing (`max()` is used as a scope join). Adding a non-lattice value there is unsafe.
2. After normalization the field would be single-valued, so it would carry no information while still participating in callable-specialization identity and hashing.
3. It invites the reading that the value names a target address space, which it does not.

The discriminator between old-style and new-style arrays is therefore the presence of an `Array` value, and the fact that storage instances have been made explicit is recorded once per kernel in `KernelState`, not once per array.

### Definition

```python
@dataclass(frozen=True)
class Array:
    """The logical and physical description of one array, without its role."""

    shape: namedisl.Set | type[auto]
    layout: ArrayLayout | type[auto]

# Narrowed alias used in annotations after the inference pass.
ResolvedArray = Array  # with both fields non-`auto`; see `Array.is_resolved`
```

`Array` deliberately contains nothing else. In particular it does **not** contain `dtype`. Reasons:

- `dtype` is shared with `ValueArg` through `KernelArgument`, and `arg.dtype` must keep meaning the same thing for every argument kind. There are roughly 85 `arg.dtype`/`tv.dtype`-style sites in `loopy/` and 349 `.dtype` sites overall; moving it would force all of them to branch on argument kind for no benefit.
- Type inference writes dtypes back with `arg.copy(dtype=...)`; keeping `dtype` at the top level leaves that untouched.
- Every layout operation that needs a dtype already takes one as a parameter: `physical_allocation(logical_shape, dtype, target)` and `runtime_interface(logical_shape, dtype, target)`.

What remains in `Array` is exactly the pair that must be kept mutually consistent — the layout must be aligned to and total on the shape — and exactly the pair a callable argument descriptor needs.

### Composition, not inheritance

Today `ArrayArg`, `ConstantArg`, `ImageArg`, and `TemporaryVariable` all inherit from `ArrayBase`. They will instead *hold* an `Array`:

```python
@dataclass(frozen=True)
class ArrayArg(KernelArgument):
    array: Array
    dtype: LoopyType | None
    alignment: int | None
    for_atomic: bool
    is_input: bool
    is_output: bool
    tags: frozenset[Tag]
    ...
```

`TemporaryVariable` holds an `Array` plus `base_storage`, `initializer`, `read_only`, and ownership/liveness-relevant state.

#### Realism assessment

This is realistic. The supporting evidence:

- `ArrayBase` appears 67 times in `loopy/` and **zero** times in `test/`, `examples/`, or `doc/`. It is effectively an internal abstraction, so removing it from the inheritance chain is not a documented public break. The public names are `ArrayArg`, `GlobalArg`, `ConstantArg`, `ImageArg`, and `TemporaryVariable`, all of which are retained.
- `GlobalArg` is *already* a function rather than a class, with the comment “Making this a function prevents incorrect use in `isinstance`.” That is the precedent this change generalizes.
- The 67 `isinstance(x, ArrayBase)` sites all ask one question — “does this thing have array-ness?” — which becomes a single helper (`get_array(x) -> Array | None`, or a `HoldsArray` protocol). That is a mechanical, low-risk substitution.

#### Payoff

Two concrete simplifications fall out, both of which reduce the number of types rather than adding to them:

- **`ImageArg` and `ConstantArg` stop being classes.** An image argument is an `ArrayArg` whose layout terminal is an `ImageLayout`; a constant argument is a read-only `ArrayArg` whose derived storage kind is `CONSTANT_BUFFER`. Both become deprecated factory *functions* alongside `GlobalArg`. The `min_target_axes`/`max_target_axes` class attributes disappear entirely, because the number of target axes is the arity of `ImageLayout`.
- **`ArrayArgDescriptor` becomes a thin wrapper over `Array`.** It currently stores `shape`, `dim_tags`, and `address_space` — which is precisely a shape, a layout, and a storage classification. After this change it holds one `Array`, so caller/callee descriptor matching and array normalization share one implementation instead of two.

#### Downsides and required care

1. **Two spellings coexist during the transition.** Forwarding properties (`ArrayArg.shape` → `self.array.shape`) are needed to keep roughly 163 `.shape` and 241 `dim_tags` sites alive, which temporarily violates the “one canonical spelling” rule. Mitigation: introduce the forwarders deprecated from day one, migrate core paths mechanically, and keep the forwarders only until the deferred transformation migration lands.
2. **`copy()` must split keyword arguments.** `arg.copy(shape=...)` is used in roughly 40 places in core plus more in transforms, and `shape` is no longer a field of `ArrayArg`. A compatibility `copy()` must route array-valued kwargs into `array` and role-valued kwargs into the role object. This is construction policy, but it lives in `copy()` rather than `__init__`, and it must round-trip correctly because `.copy(...)` may reconstruct through constructors.
3. **`Taggable` becomes ambiguous.** `ArrayBase` and `KernelArgument` are both `Taggable` today, and `ArrayArg` inherits from both with one `tags` frozenset. Decision: tags live on the *role* object only. `Array` is a plain value, not a tagged entity. Two arrays that differ only in application tags then share one `Array`.
4. **`Array` must not re-accumulate fields.** If `dtype`, `alignment`, `for_atomic`, and `offset` drift back into `Array`, it becomes `ArrayBase` under a new name and the refactor buys nothing. The rule is: `Array` is `(shape, layout)`; everything else is role.
5. **Persistent-cache and pickle formats change.** Expected and unavoidable; equality and persistent hashing become structural over the nested value, which is simpler than today's field-by-field `__eq__`.

#### Transition discriminator

During the transition, `ArrayArg.array` may be `None`, meaning “legacy array, still described by `dim_tags`/`offset`/`order`”. `array is not None` is the new-style discriminator, and the pre-codegen invariant at the end of the project is that it is never `None`. Whether storage instances have been made explicit is a separate, kernel-level fact recorded in `KernelState`.

## Logical shape

### Canonical type

```python
ArrayShape = namedisl.Set | type[auto]
ResolvedArrayShape = namedisl.Set
```

`Array.shape` may be `auto` only until the inference pass runs; every resolved array has a `namedisl.Set`. `None` is never a shape value. Legacy public entry points may accept tuple or string shapes, but must pass them immediately to a shape-construction function. Resolved internal shapes must not be represented by a `tuple | namedisl.Set` union. Keep normalization out of object constructors except for a thin backward-compatibility delegation that cannot yet be removed.

Because the canonical shape now lives at `array.shape` on a *new* type, the earlier question of whether `.shape` could change spelling in place, or needed a transitional `.index_set`, is resolved: `Array.shape` is the named set from the start, and the legacy tuple-valued `ArrayArg.shape` survives as a deprecated forwarding view returning `rectangular_shape()`. No `.index_set` transition is needed.

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

Shape stringification uses NumPy shape notation only for a **zero-based** box, for example `shape=(n, m)`, `shape=(n,)`, or `shape=()`. `namedisl.Set.is_box` is true for boxes with nonzero origins as well, so `is_box` alone is not the right predicate: printing `[n] -> { [i] : 2 <= i < n }` as `shape=(n-2,)` would name a shape that `rectangular_shape()` rejects. Non-box shapes and boxes with a nonzero origin are displayed as named sets. Reproducer and persistence formats retain the exact set.

### Shape equality and hashing

Equality is semantic after named-space alignment: `a == b` iff the two sets are equal as sets, independent of parameter order or of how their constraints were spelled.

Hashing uses a normalized representation: `coalesce → detect_equalities → remove_redundancies`, with parameters sorted by name. **This normalization is a heuristic, not a canonical form.** ISL offers no guarantee that two semantically equal sets normalize identically, so it is possible for `a == b` while `hash(a) != hash(b)`.

Spot checks against `namedisl` in this tree show the normalization to be stronger than one might fear — `0<=i<n` versus `0<=i and i<=n-1`, `0<=i<5 or 5<=i<10` versus `0<=i<10`, `exists a: i=2a` versus `i mod 2 = 0`, and `0<=i<n and 0<=j<n and i<=j` versus `0<=i<=j<n` all normalize identically — but this must not be relied upon.

The resulting invariant is therefore mandatory:

> **No correctness decision may depend on hash-based identity of a shape or a layout.** A hash disagreement between equal values must only ever cause duplicated work or a cache miss, never a divergence in generated semantics.

Consequences at the known consumers:

- *persistent caching*: a hash miss recompiles. Safe.
- *callable-specialization deduplication*: a hash miss produces two identical specializations. Wasteful, correct.
- *base-storage grouping*: if arrays are grouped into a shared base-storage allocation by hashing a storage-requirement record, a hash miss would allocate two base-storage objects where the user expected one. Still correct — it over-allocates rather than aliasing incorrectly — but surprising, and it is the one place where the grouping decision should use an explicit semantic comparison rather than a dict keyed by hash. This is called out again under “Physical extent and allocation”.

If a further consumer is added where a hash miss would be more than duplication, it must be flagged and given an explicit semantic comparison.

## Shape inference and bounds checking

Current inference already computes access ranges and then reduces them to independent minima and maxima. The new representation retains the exact access range.

For an `auto` temporary:

```text
shape = union of all relevant logical access ranges
```

Every relevant access must be represented successfully; Loopy must not silently omit an unanalyzable access from the inferred shape. An `auto` temporary with no accesses must be removed by an earlier dead-code path or diagnosed; it cannot survive as an unresolved array.

If access ranges cannot be represented quasi-affinely, inference must fail with an actionable diagnostic. Bounding-box *shape* inference is not provided: it changes the logical validity set.

### Layout inference for `auto` temporaries

Shape inference alone does not make a temporary allocatable, because allocation now comes from the layout. An inferred temporary has no user-supplied layout, so one must be chosen:

```text
shape  = exact union of access ranges
layout = make_c_layout over the bounding box of that union,
         with each axis origin taken from the box's lower bound
```

This is the one place a bounding box legitimately appears, and it is sound: a bounding-box *layout* over-allocates and never under-allocates, and it does not widen the logical validity set, so out-of-bounds accesses to the holes of a triangular shape are still rejected by bounds checking. The distinction the design insists on is exactly this one — a bounding-box shape changes logical validity, a bounding-box layout changes only allocation.

A user may override the inferred layout; `auto` shape and explicit layout is a legal combination.

### Bounds checking

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
        a lane, selector, or instance coordinate is static at this access.
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

    def map_expressions(self, mapper: ExpressionMapper) -> Self:
        """Map the layout's Pymbolic expressions and named-ISL parameters."""
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
        lane, selector, instance, and terminal-coordinate expressions.

        Layout components live in two languages: quasi-affine components are
        ``namedisl.PwAff``\\ s, composed with *index_map* directly, while
        terminal expressions are general Pymbolic expressions, for which
        composition is substitution. *index_map* must therefore be a
        single-valued map that is also convertible to per-axis Pymbolic
        expressions; a map that is not (for instance, one with unresolved
        existentially quantified variables) is rejected.

        The provider must ensure that ``f`` is injective on the view's shape;
        this method does not prove that precondition.
        """
        ...

    def validate(self, logical_shape: namedisl.Set) -> None:
        """Check structural well-formedness relative to *logical_shape*.

        Check canonical named-space alignment, component totality and declared
        ranges, wrapper-level ordering, and required allocation metadata. Do
        not attempt to prove layout injectivity; injectivity is a provider
        contract.
        """
        ...

    def lower_access(
            self, access: LogicalAccess,
            context: LayoutLoweringContext) -> LoweredAccess:
        """Lower a logical access to a storage reference and coordinate."""
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

`align_to_shape` canonicalizes a component's input space so that its set dimensions have exactly the logical shape's axis names, with no missing, extra, duplicate, or unnamed dimensions, and its parameter dimensions refer to the same names independent of ordering. This is what “named-space alignment” means here.

Concrete layouts are immutable and hashable values. `validate` is a structural check only. It verifies that components are already in the canonical named space, are defined throughout the exact logical shape, obey declared ranges such as `0 <= lane_expr < length`, respect the wrapper-level ordering below, and provide required allocation metadata. It does **not** prove injectivity.

A runtime interface describes how a host wrapper recognizes, validates, and, for outputs, allocates a concrete runtime argument. Returning `None` means that the layout does not define a host-array ABI. The array may still be usable as an internal temporary, an opaque target object, or an input accepted through custom wrapper code. Generic output allocation and standard host argument checks are unavailable without a runtime interface.

The concrete frozen records have trivial, dataclass-generated constructors. Public `make_*_layout` functions perform compatibility conversion, expression parsing, named-space alignment, normalization, and validation before constructing those records. This keeps policy out of `__init__` and makes construction logic directly testable.

### Layout types and wrapper ordering

There are six concrete layout classes and one flat union:

```python
ArrayLayout = (
    LinearLayout | RectangularLayout | ImageLayout
    | VectorLayout | SeparateLayout | InstancedLayout
)
```

An earlier draft encoded legality in the type system, using two covariant `TypeVar`s and twelve derived aliases over seven classes. That was dropped. What it encoded is a small finite rule set, the generic parameters fought the `Self`-returning protocol methods, and it did not remove the runtime check anyway — layouts are reconstructed by `.copy(...)` and deserialized from persistent caches, where static types do not apply.

Legality is instead four rules, checked once in `validate` and asserted in the factories. Wrapper levels, outermost first:

1. `InstancedLayout` — at most one, and outermost if present.
2. `SeparateLayout` — at most one.
3. `VectorLayout` — at most one.
4. terminal — exactly one of `LinearLayout`, `RectangularLayout`, `ImageLayout`.

plus:

- `ImageLayout` may not appear beneath `InstancedLayout`: images have no workgroup-local or work-item-private form.
- `ImageLayout` may not appear beneath `VectorLayout` initially; see “Image layout”.

A layout need not contain every level. A plain `LinearLayout` is a complete global layout.

### Combined layout map

Composition is defined by one abstract map over the exact logical shape `S`:

```text
L: S -> (
    storage-object selector,
    storage-instance identity,
    reuse-epoch identity,
    terminal physical coordinate,
    vector lane)
```

A terminal contributes the terminal physical coordinate. `SeparateLayout` contributes the storage-object selector. `InstancedLayout` contributes the storage-instance identity and the reuse-epoch identity. `VectorLayout` contributes the lane. Every node evaluates its component from the unchanged named logical point and passes that same point to its child.

The same logical dimension may contribute to several components. For example, with logical shape `{ [i] : 0 <= i < n }`, a packed vector layout may use `floor(i/4)` as the child's linear coordinate and `i mod 4` as the lane. This is valid because the pair is injective; there is no distinguished axis for the vector wrapper to own or remove.

**`L` is a specification, not a single computable object.** Its components live in two languages. Lanes, selectors, and instance mappings are `namedisl.PwAff`s. Terminal address expressions are general Pymbolic expressions, and they must be: loopy's ordinary strided array argument has symbolic strides — `make_temporaries_for_offsets_and_strides` replaces `lp.auto` strides with `ValueArg`s, giving `a_stride_0 * i + a_stride_1 * j`, in which a parameter multiplies an index. That is not quasi-affine, so the most common array in loopy has a terminal address ISL cannot represent at all.

Two consequences follow, and the rest of this document is built around them:

- Allocation never ranges an address expression over a set. Extents are declared or structurally derived; see “Physical extent and allocation”.
- Race and dependency analysis never composes with `L`. It works at the logical level, with layout injectivity as its soundness premise; see “Race and dependency analysis”.

### Lowered storage references and coordinates

Layout lowering produces one storage reference and a discriminated physical coordinate. Nested wrappers must not introduce competing storage names:

```python
@dataclass(frozen=True)
class StorageReference:
    name: str
    kind: StorageKind
    object_key: tuple[int, ...] | None   # compile-time SeparateLayout selector
    instance_scope: InstanceScope

class LoweredCoordinate: ...

@dataclass(frozen=True)
class LinearCoordinate(LoweredCoordinate):
    element_index: ArithmeticExpression

@dataclass(frozen=True)
class ImageCoordinate(LoweredCoordinate):
    coordinates: tuple[ArithmeticExpression, ...]

class VectorSelection: ...

@dataclass(frozen=True)
class ScalarLane(VectorSelection):
    lane: int

@dataclass(frozen=True)
class WholeVector(VectorSelection):
    """The identity permutation over all lanes."""

@dataclass(frozen=True)
class VectorCoordinate(LoweredCoordinate):
    child: LoweredCoordinate
    selection: VectorSelection

@dataclass(frozen=True)
class LoweredAccess:
    storage: StorageReference
    coordinate: LoweredCoordinate
```

A scalar vector access must lower to `ScalarLane`; whole-vector lowering produces `WholeVector`. Neither variant carries a runtime lane expression.

`StorageReference` does not carry an instance key or an epoch key. Every access is to the *current* instance and the *current* epoch — that is an enforced invariant, not an inference — so those keys are constant at every access site and contribute nothing to code generation. Liveness analysis, which does need epoch identity, obtains it from the layout and the schedule rather than from every lowered access.

Code generation dispatches on the storage reference and the coordinate variant, not on array subclasses or axis tags.

## Terminal layouts

### Linear layout

```python
@dataclass(frozen=True)
class LinearLayout(Layout):
    expr: ArithmeticExpression
    size: ArithmeticExpression
```

`expr` returns a physical element index and is evaluated in the full named logical-index environment. It may share dependencies with wrapper components; only the combined layout map must be injective.

`size` is **required** and is a separate, explicitly supplied expression giving the number of terminal elements in one storage instance. Loopy does not derive it from `expr`. This is the general rule for nonrectangular and generic storage: *the indexing expression and the sizing expression are supplied independently.* Deriving a size by ranging `expr` is not possible in general — `expr` is Pymbolic and may be non-affine — and attempting it for the affine subset would create a capability that silently disappears when a stride becomes symbolic.

The legacy `offset` attribute is not retained as independent canonical state; a base adjustment is part of `expr`.

An earlier draft allowed `size` to be a `PhysicalStorageDomain` instead of an expression. That type has been removed: a `LinearLayout` is one-dimensional by construction, so a multidimensional domain cannot describe its extent, and the only genuinely multidimensional case is `ImageLayout`, which carries its own `physical_shape`.

### Rectangular layout

Rectangular layouts are common enough to deserve a structured form:

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

`RectangularLayout` is a separate layout with computed address and allocation properties, not a subclass requiring callers to supply inherited `expr` and `size` fields. Its address expression is:

```text
base_offset + sum((logical_axis - origin) * stride)
```

The explicit origin defines how nonzero or negative logical bounds map into physical storage. Each rectangular axis reads its named logical dimension from the full environment; this is a coordinate dependency, not exclusive ownership of that dimension. Legacy `base_indices` translate to origins at the compatibility boundary.

Because the axes are structured, the extent is derived exactly from the declared `extent` and `stride` values without ranging anything: for nonnegative strides it is `base_offset + sum((extent - 1) * stride) + 1`. This is the only layout that derives its own size.

Construction conveniences are functions:

```python
make_c_layout(axes=...)
make_f_layout(axes=...)
make_strided_layout(axes=..., strides=..., origins=...)
```

The C/F factories use the rectangular physical axes supplied by the caller; they do not derive axis origins or extents from a correlated logical set. They retain enough structure for runtime wrappers to:

- validate host array shape and strides;
- infer size parameters from observed dimensions and strides;
- allocate output arrays;
- preserve existing rectangular behavior.

For a nonrectangular logical shape, a rectangular layout describes storage for a rectangular physical container. Holes in the logical set remain unused. C/F order does not imply packed triangular storage.

The first implementation supports nonnegative strides and requires the derived element index to be nonnegative within the declared physical axes. Negative strides require an explicit `LinearLayout` with a user-provided size and pointer-origin contract. A built-in rectangular layout rejects statically evident overlapping strides; symbolic cases that cannot be established from the standard C/F construction must use the explicitly trusted custom-layout path.

### Image layout

```python
@dataclass(frozen=True)
class ImageLayout(Layout):
    axis_exprs: tuple[ArithmeticExpression, ...]
    physical_shape: tuple[ArithmeticExpression, ...] | None
```

Each expression produces one image coordinate from the full named logical-index environment. `physical_shape`, when provided, supports allocation and wrapper validation. Image format, channel type, and access mode are storage metadata on the holder or the layout; they are never inferred from an address space.

Image texel channels are **not** exposed as a logical axis initially, and `ImageLayout` may not be the child of a `VectorLayout`. An earlier draft did allow this, together with channel-count validation against the image format, whole-texel read/write lowering, and an explicit rule rejecting partial image-vector writes. That was removed as new functionality rather than migration: `ImageArg` is read-only by construction today (`loopy/kernel/data.py:565`), so there is no image write path for a partial-write rule to guard; access lowering is a single hard-coded `read_imagef` with `.x` for `float32` and `as_double(.xy)` for `float64` (`loopy/target/c/codegen/expression.py:287-317`); the only tests that exercise images fix the channel order to `R` and are gated on OpenCL hardware with image support (`test/test_linalg.py:486`, `:538`); and CUDA has no image support at all.

Exposing texel channels as a logical axis remains a natural extension of this model — it is exactly `VectorLayout(child=ImageLayout(...))` with `length` equal to the format's channel count — and should be added when image writes are.

## Instanced layout

Workgroup-local storage, work-item-private storage, and sequentially reused storage share one structure. Each contributes a key to the abstract map, each drops that key from physical allocation, each justifies the drop by a disjointness argument, and each requires the same current-instance proof at every access. They differ only in *which* disjointness argument licenses the drop: hardware parallelism, or schedule sequentiality. They are therefore one node:

```python
@dataclass(frozen=True)
class HardwareAxisMapping:
    hardware_axis: int
    logical_expr: namedisl.PwAff

@dataclass(frozen=True)
class HardwareInstance:
    """Instances are disjoint because the hardware runs them concurrently."""
    scope: InstanceScope           # WORKGROUP or WORK_ITEM
    group_mappings: tuple[HardwareAxisMapping, ...]
    local_mappings: tuple[HardwareAxisMapping, ...]   # empty for WORKGROUP

@dataclass(frozen=True)
class SequentialInameMapping:
    iname: str
    logical_expr: namedisl.PwAff

@dataclass(frozen=True)
class SequentialInstance:
    """Instances are disjoint because their live ranges do not overlap."""
    mappings: tuple[SequentialInameMapping, ...]

InstanceSpec = HardwareInstance | SequentialInstance

@dataclass(frozen=True)
class InstancedLayout(Layout):
    instances: tuple[InstanceSpec, ...]
    child: ArrayLayout
```

`instances` is ordered outermost-first. At most one `HardwareInstance`, which must come first if present; at most one `SequentialInstance`. Using a tuple in one node rather than nested wrappers means the ordering rule has exactly one place to be checked.

`HardwareAxisMapping.logical_expr` maps the full logical point to the canonical zero-based hardware ID. A named logical axis or positional compatibility input is shorthand for the corresponding projection `PwAff`. Tagged inames with nonzero bases must be normalized explicitly rather than being equated directly with the zero-based hardware ID.

Semantics:

- The complete storage identity for a `WORKGROUP` hardware instance is `(group IDs, child physical coordinate)`; for `WORK_ITEM` it is `(group IDs, local/item IDs, child physical coordinate)`.
- `InstancedLayout` contributes the mapped expressions to the instance and epoch keys and passes the **unchanged** logical environment to `child`.
- Every access must prove that the mapped expressions equal the current group IDs, the current work item, and the current sequential iname values respectively. A noncurrent or unprovable access is an error. Future shuffle/communication lowering may relax the hardware case.
- The child allocation is the per-instance allocation and is not multiplied by the number of workgroups, work items, or epochs.
- In the first implementation, child physical coordinates, selectors, lanes, and allocation requirements must be independent of the instance and epoch expressions. This is an explicit uniform-allocation restriction, not axis removal.
- Inames mapped by a `SequentialInstance` must be necessarily sequential under the finalized schedule, and schedule-aware liveness must confirm that the epoch live ranges do not overlap. If either cannot be established, the layout is invalid.

### Why the instance axes are in the logical shape

Making storage instances explicit logical axes is redundant *in the first implementation*: the shape becomes a product `G × S`, only the `InstancedLayout` may read `G`, every access pins `G` to the current IDs, and allocation projects `G` away again. This is a deliberate, accepted cost, for two reasons that lie outside this document's first implementation:

1. A `compute` transformation being developed separately is significantly simpler when the instance that owns a value is an addressable coordinate rather than an implicit context.
2. Shuffles and inter-group communication — a stated non-goal *for now* — require exactly this: the ability to name a value belonging to another work item. Introducing the axes later would mean changing the shape of every local and private array after transformations have already been written against the shape.

A third, smaller benefit falls out immediately: because the instance coordinates are ordinary logical axes, race analysis gets local/private instance disjointness for free at the logical level, with no address-space-specific branching. See “Race and dependency analysis”.

## Vector layout

Vector storage is a representation wrapper, not a logical shape-axis tag:

```python
@dataclass(frozen=True)
class VectorLayout(Layout):
    lane_expr: namedisl.PwAff
    length: int
    child: LinearLayout | RectangularLayout
```

`lane_expr` is aligned to the full logical shape and contributes the lane without changing the environment passed to `child`. It must be total and satisfy `0 <= lane_expr < length` on the exact shape; `length` is a positive compile-time constant. The pair of child outputs and lane is covered by the provider's combined injectivity contract.

For a scalar access, compose `lane_expr` with the instruction-to-logical-index map and restrict it by the active code-generation domain. The result must be provably one compile-time integer, producing `ScalarLane`; dependence on a runtime parameter, unresolved piecewise branch, or nonconstant loop value is rejected.

This is a deliberate, accepted behavior change. Today a scalar vector-axis index is resolved by `eval_expr_assert_integer_constant` (`loopy/kernel/array.py:1270`), which evaluates under the unrolled code-generation context. An index that is a compile-time constant after unrolling but not quasi-affine as a function of the domain is accepted today and will be rejected under the new rule. No fallback to constant evaluation is retained.

Whole-vector access is a distinct lowering mode. It must prove that storage object, instance key, and child coordinate are invariant across the vectorized instances and that `lane_expr` is the identity over `0..length-1` across them, producing `WholeVector`. No runtime vector indexing is introduced.

Only the identity permutation is supported. An earlier draft allowed a general `StaticSwizzle(lanes: tuple[int, ...])` for reads, with a permutation requirement for writes and a target hook reporting supported compile-time swizzle forms. That was removed: today's whole-vector access is the identity — `get_access_info` emits the vector exactly when the index on a `vec` axis is literally the vectorized iname (`loopy/kernel/array.py:1365`) — and nothing in the goals above requires more. Reverse and arbitrary swizzles are a self-contained follow-up; `VectorSelection` is left as a discriminated union so that adding them does not disturb the lowering interface.

Allocation must account for target vector ABI padding. For example, an OpenCL three-vector may occupy four scalar slots. The layout reports logical vector length, while a target hook reports physical vector storage size and alignment.

Composition examples:

```python
make_vector_layout(
    lane_expr="{ [i] -> [(i mod 4)] }", length=4,
    child=make_linear_layout(expr="i // 4", size="ceil(n/4)"))
make_instanced_layout(
    instances=(HardwareInstance(scope=InstanceScope.WORKGROUP, ...),),
    child=make_vector_layout(lane_expr=..., child=make_linear_layout(...)))
make_separate_layout(
    selector_exprs=(...,),
    child=make_vector_layout(lane_expr=..., child=make_linear_layout(...)))
```

## Separate layout

Separate storage selects among distinct physical storage objects:

```python
@dataclass(frozen=True)
class SeparateLayout(Layout):
    selector_exprs: tuple[namedisl.PwAff, ...]
    child: ArrayLayout
```

Each selector expression is evaluated on the full logical point and contributes one component of the storage-object selector. The unchanged environment is passed to `child`, and the selector tuple plus child outputs is covered by the provider's combined injectivity contract.

The joint selector range must be a parameter-independent, finite, compile-time-constant Cartesian product. Selector tuples are enumerated lexicographically and use the existing deterministic subargument naming scheme. Named or positional compatibility axes become projection `PwAff`s. Sparse, correlated, or parameter-dependent selector ranges are deferred.

### Late lowering

`SeparateLayout` is **lowered late**, at code generation and ABI construction, not materialized during preprocessing. This differs from today's behavior, where preprocessing rewrites the kernel into one array per selector tuple via `ArrayArg._separation_info` (`loopy/kernel/array.py:1319-1338`).

Late lowering means:

- One logical array stays one array through shape inference, bounds checking, race analysis, and callable specialization. Those analyses never see the expansion, which is the main reason to prefer it: the IR is not rewritten behind the analyses' backs.
- `physical_allocation()` returns one `PhysicalStorageObject` per selector tuple, each with its `object_key` set. Argument-list and declaration expansion therefore happens at *signature generation* time, driven by the allocation result, rather than by rewriting instructions.
- At each access, composing the selectors with the access map must yield one compile-time selector tuple. Code generation uses it to pick the object name, which becomes `StorageReference.object_key` and `StorageReference.name`. A non-constant selector is an error unless a target-specific indirect-object mechanism is introduced later.
- Runtime wrappers present one host argument per physical object, reusing the existing deterministic naming scheme. The mapping from selector tuples to physical names is lowering output, not canonical array metadata.
- At a call boundary, a `SubArrayRef` may either fix the selector — specializing to one fiber, which preserves injectivity on that fiber — or sweep it, in which case the callee descriptor retains the `SeparateLayout` and the call passes all objects.

Late lowering is what makes `object_key` on `PhysicalStorageObject` and `StorageReference` load-bearing; under early materialization they would be dead by the time code generation ran.

`make_separate_layout(..., child=make_vector_layout(...))` — separate arrays whose entries are vectors — is permitted. The wrapper-ordering rules exclude the reverse order and repeated wrappers. No restriction is placed on which logical dimensions contribute to the selectors, lane, and child coordinates.

## Injectivity contract

Plain physical-address injectivity is incompatible with per-instance allocation and sequential storage reuse. The contract applies to the complete combined map `L` on the exact logical shape, after named-space alignment:

> The tuple of storage-object selector, storage-instance key, reuse-epoch key, terminal coordinate, and vector lane uniquely identifies a logical point.

Equivalently:

```text
L(x) = L(y)  implies  x = y
```

The instance and epoch keys make the abstract map injective even though allocation deliberately drops them and reuses physical storage.

The property the analyses actually consume is the *physical* one, which does not follow from the abstract contract alone. It is the abstract contract **plus** the hardware-concurrency and liveness arguments that justifies dropping those keys:

> Restricted to logical points that are concurrently live, the tuple of storage-object selector, terminal coordinate, and vector lane uniquely identifies a logical point within one storage instance.

The vector lane must be part of that tuple: two distinct lanes of one vector value share a storage object and terminal coordinate, and are distinguished only by the lane.

**This contract is load-bearing for correctness, not merely for allocation sanity.** Race analysis operates at the logical level (see below) and is sound *only because* distinct logical points map to distinct storage. A provider that supplies a noninjective layout does not merely over-allocate; it silently invalidates race and dependency analysis. This is why noninjective layouts are a non-goal rather than a degraded mode.

Injectivity is a semantic contract on every layout provider, not something `validate` attempts to prove. This remains true when all components are quasi-affine: Loopy checks that component expressions are well-formed, total, and in range, but it does not run a general two-copy collision query. A component may be noninjective by itself — `floor(i/4)` and `i mod 4` are each noninjective while their pair is injective — so local checks on individual fields would not establish the contract anyway. Built-in factory documentation states why its standard constructions satisfy it; users supplying custom expressions are responsible for the combined map. An optional diagnostic injectivity checker may be added later without becoming part of normal validation.

Pullback through a subarray/reindexing map preserves the contract only when that map is injective on the new logical shape. `pullback` therefore has injective reindexing as a precondition; Loopy rejects mappings that are statically known to repeat elements but does not promise a general proof. Specializing a separate selector preserves injectivity on that selector fiber.

## Physical extent and allocation

Logical shape says which logical values exist; it does not say what to allocate. Allocation planning converts a shape and layout into descriptions of physical storage objects. It does not multiply those descriptions by the number of runtime workgroups, work items, or epochs.

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
    element_extent: ArithmeticExpression
    image_shape: tuple[ArithmeticExpression, ...] | None
    alignment: int | None
```

`object_key` is the compile-time selector tuple of a `SeparateLayout`; it is `None` for an unseparated array. `element_extent` describes one storage instance and is the number of terminal storage elements; it never includes the number of workgroups, work items, separate objects, or sequential epochs. `image_shape` replaces it for `IMAGE` storage. `alignment` is a byte alignment, or `None` when the target/default ABI decides it.

`instance_scope` is not a field: it is derived from the layout, as described under “Storage kind”.

### From a layout to an allocation

1. Enumerate the finite storage-object selector values. An unseparated layout has one; a `SeparateLayout` has one per physical object.
2. Take the per-instance extent from the terminal layout. `LinearLayout` supplies it explicitly as `size`. `RectangularLayout` derives it structurally from declared origins, extents, and strides. `ImageLayout` supplies `physical_shape`. **No extent is ever obtained by ranging an address expression over a set.**
3. Check that the requirement is uniform across instance keys. The first implementation rejects a layout whose per-instance extent varies by workgroup, work item, or epoch.
4. Apply vector ABI rules. A vector changes the terminal element type, size, and alignment; its lanes do not create additional objects.
5. Derive the storage kind and instance scope from the layout. Runtime execution supplies one instance at global, workgroup, or work-item scope as appropriate; the allocation descriptor itself is not replicated.

Examples:

- A global linear temporary yields one `GLOBAL_BUFFER` object with `GLOBAL` instance scope.
- A local temporary yields one `LOCAL_MEMORY` object description; each workgroup receives an instance with the stated per-instance extent.
- A private temporary yields one `PRIVATE_MEMORY` description, one instance per work item.
- A separate layout yields several object descriptions distinguished by `object_key`.
- A `SequentialInstance` adds no object and no physical instance; its epochs reuse the enclosing instance.

`TemporaryVariable.storage_shape`, `base_indices`, and `offset` do not remain canonical public state:

- lower bounds belong in the logical shape;
- offsets belong in the terminal layout expression;
- explicit storage sizing belongs in the layout.

When several arrays share base storage, compatibility includes object key, storage kind, instance scope, alignment, dtype, and extent — not merely the largest element count. Per the hashing invariant above, base-storage grouping must compare these requirements semantically rather than keying a dictionary on their hash: a hash disagreement between equal requirements would split one intended base-storage allocation into two. That over-allocates rather than aliasing incorrectly, so it is safe, but it is surprising and should be avoided by construction.

## Explicit storage instances

`make_storage_instances_explicit` is a translation-unit transform and must be idempotent. It converts old-style local and private arrays into arrays whose owning instance is an addressable logical coordinate. Successful completion is recorded in `KernelState`, not on each array.

### Global conversion

```text
old shape: S
old layout: f(i)
new Array: (S, terminal/representation layout f(i))
```

An unwrapped terminal or representation layout has `GLOBAL_BUFFER` storage kind by default, or `CONSTANT_BUFFER` for a read-only array so requested. Array arguments and temporaries differ in ownership and liveness, not in layout.

### Local conversion

```text
old access: A[i...]
new access: A[g0, g1, ..., i...]
new shape: group domain x old shape
new layout: make_instanced_layout(
    instances=(HardwareInstance(WORKGROUP, group_mappings=...),),
    child=old physical layout)
```

### Private conversion

```text
old access: A[i...]
new access: A[g0, ..., l0, ..., i...]
new shape: group domain x local domain x old shape
new layout: make_instanced_layout(
    instances=(HardwareInstance(WORK_ITEM, group_mappings=..., local_mappings=...),),
    child=old physical layout)
```

Canonical group/item coordinates must match code generation, including hardware inames with nonzero bases.

### Sequential conversion

Sequential private axes are present in the logical shape and every access uses the current iname value. A `SequentialInstance` records that the physical allocation may be reused, subject to liveness.

The transform creates named instance dimensions and projection `PwAff`s for their mappings; later layout lowering does not depend on those dimensions occupying a tuple prefix. Instance axes for images and multiple hardware instances are rejected.

### Validation

For every access, Loopy must prove:

- local group coordinates equal current group IDs;
- private group and item coordinates equal the current work item;
- sequential coordinates equal current sequential iname values.

Failure or inability to prove equality is an error. Future shuffle/communication lowering may relax the hardware cases.

## Runtime wrappers

Runtime wrappers validate physical storage, not the logical shape directly.

Layouts may expose a runtime interface:

```python
@dataclass(frozen=True)
class RuntimeArrayInterface:
    physical_shape: tuple[ArithmeticExpression, ...] | None
    strides: tuple[ArithmeticExpression, ...] | None
    byte_size: ArithmeticExpression | None
    alignment: int | None
    equations: tuple[ParameterEquation, ...]
```

An array “has no runtime interface” when `runtime_interface(...)` returns `None`. Such a layout has no standard host-array contract. This does not make it invalid: device-only temporaries, opaque target objects, and custom execution wrappers may not need one. It means the standard wrapper cannot infer parameters from that argument, validate its shape/strides beyond separately supplied byte-size information, or allocate it as an output.

Rectangular layouts must continue to support:

- validation of runtime rank, dimensions, and strides;
- inference of integral size parameters from runtime arrays;
- output allocation;
- existing singleton- and empty-axis stride rules where applicable;
- `skip_arg_checks` behavior.

### Parameter inference

Parameter inference is expressed as equations contributed by the layout. A physical extent `n + 2`, for example, contributes an equation against the observed runtime dimension.

This is a refactor of existing machinery, not new infrastructure. `loopy/target/execution.py:185-320` already builds `_ArgFindingEquation` records with an `order` field for dependency staging and a `based_on_names` field enforcing “do not use more than one fact from each array”, and solves them with `pymbolic.algorithm.solve_affine_equations_for`. The change is to move equation *construction* from hard-coded inspection of `arg.shape` and `get_strides(arg)` into `RuntimeArrayInterface.equations`, leaving the grouping, ordering, and solving code as it stands.

For generic layouts:

- the declared size permits a one-dimensional internal allocation, but does not by itself define a host-array ABI;
- wrappers may validate byte size and alignment if a runtime interface is supplied;
- shape-parameter inference is available only when the layout provides equations;
- output allocation requires an explicit runtime physical interface;
- ambiguous inference or allocation must produce a targeted error, not a guessed layout.

Logical bounds remain compile-time/polyhedral checks and are separate from runtime physical-storage checks.

## Code generation

Code generation assumes every array holds a resolved `Array` and that storage instances are explicit.

### Access lowering

Replace `get_access_info` with layout-driven lowering. The access is represented by named expressions plus an instruction-domain-to-logical-index map. Every quasi-affine layout component is composed with that map and restricted by the active code-generation domain. Linear, image, vector, and separate accesses are explicit lowered variants. Offsets and target-axis accumulation are not independently reapplied by code generators.

Scalar vector lanes and separate selectors must reduce to compile-time singleton values. Whole-vector lowering must prove child-coordinate invariance and an identity lane permutation.

### Declarations

Declarations derive from the layout's storage kind:

| Storage kind | OpenCL | CUDA |
|---|---|---|
| `GLOBAL_BUFFER`, argument | `__global T *` | pointer kernel parameter |
| `GLOBAL_BUFFER`, persistent temporary | host/device allocation passed to kernels | device allocation passed to kernels |
| `CONSTANT_BUFFER` | `__constant T *` | `__constant__` |
| `LOCAL_MEMORY` | `__local` allocation | `__shared__` allocation |
| `PRIVATE_MEMORY` | automatic storage | thread-local/register-backed automatic storage |
| `IMAGE` | image object and image intrinsics | unsupported until texture/surface support exists |

Atomics and volatile casts query the derived storage kind, not an address space.

### Pre-codegen invariants

- Every array holds a resolved `Array`; no `auto` shape or layout remains.
- Storage instances are explicit, as recorded in `KernelState`.
- Every logical access has the correct named space.
- Every layout `PwAff` is aligned with the canonical logical shape, total on that shape, and within its declared range.
- Every layout satisfies the wrapper-ordering rules.
- Every instanced access is current-instance and current-epoch legal.
- Every terminal layout has a declared or structurally derived extent when Loopy allocates it, and instance fibers have uniform requirements.
- No unlowered legacy dim tags remain.
- Every scalar vector lane and separate selector is a compile-time singleton.
- Every whole-vector access has a proved child-coordinate-invariant identity lane permutation.

## Race and dependency analysis

**Race, dependency, and ordering analysis operates at the logical level only.** It does not compose accesses with the combined layout map, does not compute physical coordinates, and has no notion of a physical access footprint.

This is possible because of the injectivity contract: distinct logical points map to distinct storage within one storage instance. Logical disjointness therefore implies physical disjointness, and the analysis can work entirely in the language it already speaks — access ranges as `namedisl` sets over instruction domains, which are quasi-affine by construction.

It is also *necessary*: the physical coordinate of loopy's most common array — a strided argument with symbolic strides — is not quasi-affine (see “Combined layout map”), so a physical-level collision query would be unavailable exactly where it is most needed.

The analysis becomes:

```text
execution coordinates -> array name + logical index set
```

with a two-copy collision query over logical indices:

- duplicate the execution coordinates;
- require a difference in at least one relevant concurrent coordinate;
- require the same array name;
- require overlapping schedule-derived live ranges;
- require intersecting logical index sets;
- test nonemptiness.

This replaces the current syntactic assumption that mentioning a parallel iname in a subscript proves injectivity. Expressions such as `i % 2` and `i - i` are handled correctly because the query is an actual collision test, not an occurrence check.

Three things follow directly from the representation, with no special-case logic:

- **Instance disjointness is free.** Because group and item coordinates are ordinary logical axes, two accesses from different workgroups to a local array differ in the group axis, so their logical index sets do not intersect. No address-space branching is required, and `_is_racing_iname_tag` and the local/global-specific race paths can be removed.
- **Sequential epochs do not race.** Distinct epoch coordinates are distinct logical points, and the epochs are sequential by construction.
- **Whole-vector accesses cover a set.** A whole-vector access is a *set* of logical points — all lanes — rather than a point. The existing access-range machinery already produces sets, so a whole-vector access correctly overlaps each of its constituent lanes with no footprint model.

### Base storage

Distinct arrays sharing `base_storage` have different logical namespaces, and the analysis does **not** reason about their layouts in detail. Two distinct array names sharing base storage are conservatively assumed to overlap whenever their live ranges do. Proving disjointness there would require composing two layouts into a common physical coordinate, which is exactly the non-affine computation this design avoids.

This preserves today's conservative behavior. A future extension could prove disjointness for the restricted case where both layouts are rectangular with compile-time strides, but it is not part of this design.

### Atomics

Target-specific atomic granularity is a code-generation concern, handled conservatively where it exceeds one element. It is not modeled in the analysis.

## Calls to callable kernels

Array argument descriptors hold an `Array`:

```python
@dataclass(frozen=True)
class ArrayArgDescriptor:
    array: Array
```

This is the same value the caller's argument holds, so descriptor matching and array normalization share one implementation. The old `address_space` field is removed: storage classification is derived from the layout. Any callable-specialization state that does not yet know a shape or layout uses a separate unresolved descriptor type; it does not encode that state as `shape=None` or `layout=None`.

For a `SubArrayRef`:

1. Obtain the exact swept-iname domain.
2. Build a named map from callee-visible logical indices to source logical indices.
3. Define the callee shape as the swept domain, and *check* that it is contained in the exact preimage of the source shape. Do not define it as the intersection of the two: intersecting would silently discard the part of the swept domain that lies outside the source array, turning an out-of-bounds subarray reference into a valid, smaller one. The containment check is exactly the bounds check for the reference.
4. Pull back every source-layout component, including `PwAff` lanes, selectors, and instance mappings, through this map.
5. Preserve correlated and union domains and translate parameter namespaces explicitly.
6. Fix nonswept storage-instance dimensions to current group/item/epoch values.
7. Require the reindexing map to be injective on the callee shape and reject statically evident repeated-element views; `pullback` does not itself prove this precondition.
8. Return the resulting `Array` with normalized named spaces.

A local view remains tied to the current group. A private view remains tied to the current item. Calls that imply another storage instance are invalid. No implicit global/local/private conversion occurs at a call boundary.

Existing callable specialization behavior is retained: the concrete caller descriptor specializes the callee.

Third-party `InKernelCallable.with_descrs` implementations may inspect tuple shapes and dim tags directly. Provide migration helpers:

```python
descr.array.rectangular_shape()
descr.array.linear_strides()
descr.array.layout
```

and document the source-level compatibility break.

## Generated subkernels and liveness

For device programs split at global barriers, persistence is derived rather than stored in the layout:

- schedule-aware liveness determines whether a value is live across a generated-subkernel boundary;
- a live `GLOBAL_BUFFER` or `CONSTANT_BUFFER` value may use a host/device allocation passed between generated subkernels;
- `LOCAL_MEMORY` and `PRIVATE_MEMORY` instances physically cannot survive a launch boundary, so a value live across that boundary requires an explicit save/reload transformation or is rejected;
- allocation and release points for global temporaries follow their computed live ranges;
- reuse proposed by a `SequentialInstance` is legal only when the relevant epoch live ranges are nonoverlapping.

Queries currently based on `AddressSpace.GLOBAL` or `.LOCAL` must instead combine derived storage kind and instance scope with liveness and ownership information. This includes temporary passing, local-memory accounting, base-storage checks, and host-side global temporary allocation.

## Compatibility policy

### Retained inputs

Initially retain:

- legacy top-level `shape=None` only at entry points that immediately translate it to supported exact `auto` inference;
- tuple shape input;
- string shape input;
- `dim_names` input;
- `strides`, `order`, and `dim_tags` input;
- `offset`, `base_indices`, and `storage_shape` input;
- old address spaces as construction-time requests;
- `tag_array_axes`;
- `ImageArg` and `ConstantArg` as factory functions.

These are conversion front ends, not canonical stored state. Unsupported `shape=None` uses are rejected, and `None` never reaches `Array.shape` or an array argument descriptor.

### Compatibility properties

- `ArrayArg.shape` and `TemporaryVariable.shape` survive as deprecated forwarding properties returning `rectangular_shape()`, raising for nonrectangular shapes.
- A computed `.dim_tags` compatibility view may be offered for layouts exactly representable by old tags. It must return no misleading answer for arbitrary layouts. A kernel using a layout feature with no old-tag spelling has no legacy code-generation path; during the transition, new layout features are available only on the new path, and the compatibility suite must not require them on the old one.
- `copy()` on an array holder accepts both array-valued and role-valued keyword arguments, routing the former into `array`.
- Breaking compatibility for canonical `storage_shape`, `base_indices`, and `offset` state is acceptable; legacy constructor values are translated where inexpensive.

### `tag_array_axes`

`tag_array_axes` parses legacy syntax and replaces the array layout:

- C/F/nesting tags call the rectangular-layout factories;
- fixed strides call `make_strided_layout`;
- `vec` constructs the selected-axis projection `PwAff` and calls `make_vector_layout`;
- `sep` constructs selector projection `PwAff`s and calls `make_separate_layout`;
- new shorthands may construct instance projection mappings or image coordinates.

Add a direct `set_array_layout` API for new code.

## Remaining prototype decisions

1. The division between target-independent vector metadata and the target ABI hook implementation.
2. The richness of optional generic runtime interfaces beyond byte size and alignment.
3. Whether `ImageArg`/`ConstantArg` factory functions should warn immediately or after one release.

Settled: logical axes are canonically named; all layout nodes retain the full named environment; the layout union is flat with wrapper ordering checked in `validate`; separate layouts are lowered late; storage kind and instance scope are derived; generic extents are declared rather than ranged; race analysis is logical-level only; `Array` is `(shape, layout)` and is held rather than inherited.
