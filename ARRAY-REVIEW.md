# Review of `ARRAY-DESIGN.md` / `ARRAY-IMPL.md`

> **Status: superseded.** This review was written against the draft at commit
> `acf3fde6`. Its findings have been dispositioned and `ARRAY-DESIGN.md` and
> `ARRAY-IMPL.md` have been rewritten accordingly. It is retained only as a
> record of the reasoning. Where it disagrees with the current design
> documents, the design documents win.
>
> Disposition:
>
> | # | Outcome |
> |---|---|
> | A1 | **Rejected.** The explicit instance axes stay. They are needed by a separate `compute` transformation and by eventual shuffle/inter-group communication; the redundancy in v1 is an accepted, documented cost. |
> | A2 | **Adopted, and extended.** No new address space — but rather than a `KernelState` flag alone, the new semantics become a new `Array` type that arguments and temporaries *hold* rather than inherit. |
> | A3 | Adopted — one `InstancedLayout` with an ordered tuple of instance specs. |
> | A4 | Applied in place. |
> | A5 | Adopted — protocol/ABC ambiguity resolved. |
> | B1 | Adopted — image-backed vectors and all texel/channel machinery cut. |
> | B2 | Adopted — identity permutation only; swizzle target hook cut. |
> | B3 | **Adopted in spirit, inverted in method.** `SeparateLayout` stays in the layout algebra, and `sep` is *late*-lowered rather than materialized in preprocessing. This makes `object_key` and the separate-aware lowering paths load-bearing rather than dead. |
> | B4 | Adopted — flat union plus wrapper-ordering rules in `validate`. |
> | B5 | Adopted — `PhysicalStorageDomain` deleted; `StorageKind`/`InstanceScope` derived. |
> | C1 | **Adopted, and taken further.** Race analysis is logical-level *only*: no physical composition, no footprint model, `base_storage` conservative by design. Non-rectangular arrays supply sizing as an expression separate from the indexing expression. |
> | C2 | Adopted — `auto` temporaries get a bounding-box *layout* over the exact union shape. |
> | C3 | **Adopted with a correction.** Hashing may indeed return unequal hashes for equal sets; this is now stated as a permanent property rather than something to engineer away, with an invariant that no correctness decision may depend on hash identity, and `base_storage` grouping flagged as the one place needing explicit semantic comparison. |
> | C4 | **Deferred.** The transform inventory is recorded in `ARRAY-IMPL.md` under "Deferred: transformation migration"; no detailed plan yet. |
> | C5 | Vector-lane regression accepted (no fallback retained). Parameter inference refactor adopted. Implementation plan intentionally left coarse. |

Review axes as requested: mathematical consistency, Occam's razor, realizability.

The core of the proposal is sound and is the right direction: a logical index set
separated from an injective logical-to-physical layout, with allocation, bounds,
and race analysis all reading the same two objects. The criticism below is almost
entirely about *scope*: several mechanisms are introduced that either carry no
information, duplicate information already present elsewhere, or are new features
rather than migrations of existing ones.

Findings are ordered by how much they would change the design.

---

## Summary of recommendations

| # | Recommendation | Axis | Size of cut |
|---|---|---|---|
| A1 | Drop the added group/item logical dimensions; keep hardware scope as an unindexed wrapper | consistency / Occam | large |
| A2 | Drop `AddressSpace.UNIVERSAL`; use `KernelState` and `layout is not None` | consistency / Occam | medium |
| A3 | Unify `LocalLayout`/`PrivateLayout`/`InamePrivateLayout` into one instanced-layout node | Occam | medium |
| B1 | Cut image-backed `VectorLayout` and all texel/channel machinery from v1 | Occam | large |
| B2 | Cut general static swizzles; require the identity permutation in v1 | Occam | medium |
| B3 | Keep `sep` as today's preprocessing expansion; drop `SeparateLayout` from the layout algebra | Occam | medium |
| B4 | Replace the 12-alias generic type algebra with a flat union plus a `validate` rule | Occam | medium |
| B5 | Delete `PhysicalStorageDomain` and derive `InstanceScope`/`StorageKind` | Occam | small |
| C1 | State plainly that there is no single ISL map `L`; invert the emphasis in race analysis | realizability | — |
| C2 | Say how an `auto` temporary gets its *layout* | gap | — |
| C3 | Define shape `__eq__` on the normalized form, not `.equals()` | realizability | — |
| C4 | Schedule `loopy/transform/` in the implementation plan | realizability | large |

---

## A. Mathematical consistency

### A1. The instance dimensions added by universalization are provably redundant

This is the largest consistency problem.

Universalization rewrites a local array of shape `S` to shape `G × S` and a
private array to `G × L × S`, and then requires (under *Validation*, and again
under each scope wrapper) that **every** access prove its group/item coordinates
equal the current group/item IDs.

A dimension whose value is forced, at every access, to equal a value supplied by
the execution context is not a logical index. It is a storage-instance selector.
The design then says so itself twice over:

- `LocalLayout.group_mappings` maps the logical point back to the hardware ID —
  so the same information is stored a second time, as a `PwAff` that projects out
  the dimension that was just added;
- the first implementation "requires child physical coordinates, selectors, lanes,
  and allocation requirements to be independent of the group-instance expressions",
  i.e. the layout below the wrapper is required to ignore `G` entirely.

So in v1 the shape is a product `G × S` in which nothing but the scope wrapper may
read `G`, every access pins `G` to a constant, and allocation immediately projects
`G` away again. The net information content of the added dimensions is zero.

Consequences of keeping them:

- `num_axes`, `axis_names`, and `rectangular_shape()` change meaning mid-pipeline,
  so no user-facing shape query is stable across preprocessing;
- bounds checking gains dimensions whose constraints are satisfied by construction;
- an entire pass is needed to rewrite every access, predicate, substitution, and
  `SubArrayRef` (Phase 6 tasks 6, 7, 9, 10), plus the "fix nonswept storage-instance
  dimensions to current values" step at every call boundary;
- `HardwareAxisMapping` and nonzero-hardware-iname-base normalization exist only to
  undo the addition.

**Recommendation.** Keep hardware scope as an *unindexed* wrapper:

```python
@dataclass(frozen=True)
class ScopedLayout:
    instance_scope: InstanceScope   # WORKGROUP or WORK_ITEM
    child: ElementRepresentationLayout
```

meaning "one instance per workgroup / per work item, always accessed from the
current instance". This is exactly the semantics the design already mandates, and
it is what loopy's local/private temporaries mean today.

The stated motivation is race analysis ("local access from different groups → no
alias"). That is fully preserved: the race query already duplicates *execution*
coordinates, and group IDs are execution coordinates. Two accesses can alias only
if they agree on the execution coordinates at or above the instance scope. You need
group IDs in the execution coordinate space, which you have; you do not need them in
the array's logical index space.

What is genuinely lost is cross-instance addressing — reading another work item's
private value via a shuffle. That is an explicit **non-goal** of this document
("Remote local/private access through shuffles or communication is not initially
supported"). When it is wanted, the explicit instance dimensions and the
`HardwareAxisMapping` come back, and that is the right time to pay for them. The
design should either make this case (and defer the dimensions) or state why the
redundancy is worth carrying now.

### A2. `AddressSpace.UNIVERSAL` carries no information

After normalization every array's address space is `UNIVERSAL`. A field with one
value is not a field. What the design actually needs is:

- *for code generation*: the storage kind and instance scope — which come from the
  layout;
- *for phase discipline*: a marker that universalization has run — which is a
  property of the kernel/translation unit, not of each array.

Putting it in `AddressSpace` is actively harmful in three ways:

1. `AddressSpace` is an `IntEnum` whose ordering is load-bearing (`max()` as a scope
   join). `AGENTS.md` calls out adding a non-lattice value there as unsafe. The design
   acknowledges this and then does it anyway, with "replace the `max()` joins first"
   as the mitigation. Not creating the hazard is cheaper than sequencing around it.
2. `ArrayArgDescriptor.address_space` becomes a constant field that still participates
   in callable specialization identity and hashing, which `AGENTS.md` flags as
   hash-sensitive. It is a constant contribution to a hash that exists to distinguish
   specializations.
3. It invites the reading that `UNIVERSAL` is a target address space, which the design
   has to spend a paragraph denying.

**Recommendation.** Do not add the enum value. Record universalization in
`KernelState` (or a translation-unit flag), let `layout is not None` discriminate
migrated arrays during the transition, and delete `address_space` from
`ArrayArgDescriptor` once layouts are canonical. Retain `AddressSpace` strictly as
legacy input. This deletes the "*Universal address space*" section, the enum-ordering
caveat, and Phase 6 task 1–2 friction — while keeping the transform, which is where
the real work is. Consider renaming the transform to something like
`normalize_array_storage`, since "universal address space" would no longer name
anything.

### A3. The three scope wrappers are one concept

`LocalLayout`, `PrivateLayout`, and `InamePrivateLayout` have identical structure:

| | key contributed | dropped for allocation because | proved current at each access |
|---|---|---|---|
| `LocalLayout` | group IDs | instances are hardware-disjoint | yes |
| `PrivateLayout` | group + item IDs | instances are hardware-disjoint | yes |
| `InamePrivateLayout` | epoch values | live ranges are schedule-disjoint | yes |

Each contributes a key to the abstract map, each drops that key from physical
allocation, each justifies the drop by a disjointness argument, and each requires the
same current-instance proof. They differ only in *which* disjointness argument
licenses the drop.

**Recommendation.** One node:

```python
@dataclass(frozen=True)
class InstancedLayout:
    # hardware disjointness (WORKGROUP/WORK_ITEM) or schedule disjointness (inames)
    justification: HardwareInstance | SequentialReuse
    child: ElementRepresentationLayout
```

One validation rule, one allocation rule (drop the key, require uniformity), one
pullback rule, one current-instance check. `PrivateLayout` nesting inside
`LocalLayout` also stops being a type-level special case and becomes an ordinary
value-level check on the justification.

### A4. Nits fixed in place

Applied directly to `ARRAY-DESIGN.md`:

- **Injectivity, concurrently-live form.** The sentence "equality of storage object,
  instance identity, and physical coordinate implies equality of the logical point"
  omitted the representation coordinate, which makes it false — two lanes of one
  vector agree on object, instance, and terminal coordinate. Also, the physical
  property does not *follow* from the abstract contract; it follows from the contract
  **plus** the liveness proof. Both now stated explicitly.
- **`is_box` is the wrong predicate for NumPy-shape display.** `is_box()` is true for
  boxes with nonzero origins (verified against `namedisl`), so
  `[n] -> { [i] : 2 <= i < n }` would print as `shape=(n-2,)` — a spelling that
  `rectangular_shape()` rejects. Display now requires a zero-based box.
- **`StorageKind` was missing the constant address space.** `ConstantArg` and
  initialized read-only temporaries lower to `__constant` (OpenCL) /
  `__constant__` (CUDA), with their own pointer qualifier, hardware path, and
  capacity limits. That is a storage kind, not a form of ownership, and the design
  filed it under ownership. `CONSTANT_BUFFER` added.
- **`num_axes` on an `auto` shape.** `auto` has no rank, and the design elsewhere
  forbids representing unknown rank. Now stated that `num_axes`/`axis_names` raise.
- **`SubArrayRef` callee shape.** "Exact preimage of the source shape, *intersected*
  with the swept domain" silently converts an out-of-bounds subarray reference into a
  valid smaller one. Changed to: callee shape *is* the swept domain, with containment
  in the preimage *checked* — which is precisely the bounds check for the reference.
- **`pullback` cannot compose a Pymbolic terminal with an ISL map.** Noted that the
  reindexing must be single-valued *and* convertible to per-axis Pymbolic expressions.

### A5. Remaining consistency nit (not fixed, needs a decision)

`Layout` is declared a `Protocol` but concrete layouts are written
`class LinearLayout(Layout)`. Pick one: structural (no inheritance, `Protocol`) or
nominal (`ABC`). Mixing them means `isinstance` checks silently do something
different from what the annotation suggests, and `Self`-returning protocol methods
under `@dataclass(frozen=True)` inheritance are a known source of type-checker noise.

---

## B. Occam's razor

### B1. Image-backed vectors are new functionality, not a migration

The design devotes substantial machinery to image texel channels:
`ImageVectorLayout`, `SeparateLayout[ImageVectorLayout]`, channel-count validation
against the image format, whole-texel read/write lowering, rejection of partial
image-vector writes with an explicit "do not synthesize read-modify-write" rule, a
`VectorCoordinate(child=ImageCoordinate(...))` lowering path, and a pre-codegen
invariant. Phases 3, 4, 7, 9, and 11 all carry image-vector tasks.

Current state of image support in tree:

- `ImageArg` is **read-only** by construction (`loopy/kernel/data.py:565`,
  "ImageArg cannot be an output (for now)"). There is no image write path at all.
- Access lowering is one hard-coded `read_imagef` with `.x` for `float32` and
  `as_double(.xy)` for `float64` (`loopy/target/c/codegen/expression.py:287-317`).
- Channel order is fixed to `R` in the only tests that exercise it
  (`test/test_linalg.py:486`, `:538`), both gated on OpenCL hardware with image support
  and skipped on pocl.
- CUDA has no image support.

So partial-write rejection is a rule about a code path that does not exist, and
channel-count validation guards a feature (multi-channel texels as a logical axis)
that has never been supported.

**Recommendation.** For v1, keep `ImageLayout` as a terminal only, exclude
`ImageLayout` as a `VectorLayout` child, and drop every channel/texel rule. This
removes one type parameter case, two aliases, three pre-codegen invariants, a
factory validation rule, and roughly a dozen planned test cases, while preserving
100% of current behavior. Note the intended extension in a "future work" section.

### B2. General static swizzles are unnecessary

The design introduces `StaticSwizzle(lanes: tuple[int, ...])`, requires whole-vector
writes to prove the tuple is a permutation of `0..length-1`, and adds a **target hook
for supported compile-time swizzle forms** (Phase 0 item 6, Phase 3 task 13, Phase 7
tests for "static identity, reverse, and arbitrary supported read swizzles"). It is
also one of the five listed *Remaining prototype decisions*.

Today's whole-vector access is the identity: `get_access_info` emits the vector
whenever the index on a `vec` axis is literally the vectorized iname
(`loopy/kernel/array.py:1365-1373`). Nothing in the stated goals requires more.

**Recommendation.** Keep `StaticSwizzle` in the lowered-access data model (it costs
nothing and documents intent), but in v1 accept and emit only the identity
permutation. Delete the target capability hook and remaining-decision #3. Reverse and
arbitrary swizzles are a separate, self-contained follow-up.

### B3. `SeparateLayout` is erased before it is used

The design chooses **early materialization**: preprocessing enumerates the selector
tuples, creates one physical argument per tuple, specializes the child, and
*replaces `SeparateLayout` with the specialized child*. Late lowering is explicitly
"a possible future extension, not an alternative in the initial implementation".

So `SeparateLayout` never reaches code generation. Yet it is carried through:

- a covariant `SeparateChildT` type parameter and four derived aliases;
- a "storage-object selector" component in the combined map;
- `object_key` in `PhysicalStorageObject`;
- an object selector in `StorageReference`;
- "distinct separate selectors prove disjointness" in race analysis;
- a pre-codegen invariant about separate storage;
- Phase 7 task 10 and Phase 9 task 10.

All of that is unreachable in v1, because by codegen time each fiber is an ordinary
array. And loopy already has exactly this mechanism:
`ArrayArg._separation_info` with `subarray_names` keyed by the selector tuple
(`loopy/kernel/array.py:1319-1338`).

**Recommendation.** Keep `sep` where it is: a front-end tag that preprocessing
expands, expressed in the new world as a *construction-time* option on the array
rather than a member of the layout algebra. Add `SeparateLayout` to the layout
algebra when and only when late lowering (indirect object selection) is actually
implemented. This deletes roughly one third of the type algebra and the selector
component of the combined map.

### B4. The generic type algebra costs more than it buys

The design defines twelve type aliases and two covariant `TypeVar`s
(`ElementTerminalLayout`, `TerminalLayout`, `VectorChildLayout`, `VectorChildT`,
`ElementVectorLayout`, `ImageVectorLayout`, `AnyVectorLayout`, `SeparateChildLayout`,
`SeparateChildT`, `RepresentationLayout`, `ElementSeparateLayout`,
`ElementRepresentationLayout`, `InstanceScopedChildLayout`, `ArrayLayout`) to encode
legality over **seven** concrete classes.

What it encodes is a small finite rule set: a total order on wrapper levels
(scope → sequential reuse → representation → terminal), at most one wrapper per
level, and "images may not be scoped". That is three sentences and one loop in
`validate`.

Costs of the type-level encoding:

- generic covariant layouts fight `Self`-returning protocol methods (`map_expr`,
  `pullback`, `align_to_shape`) — `VectorLayout[X].pullback()` returning `Self`
  cannot be expressed cleanly;
- every future wrapper multiplies the alias count;
- it does not remove the runtime check, because layouts are reconstructed by
  `.copy(...)` and deserialized from persistent caches, where static types do not
  apply. The design already says factories must "reject dynamically typed attempts".

With B1 and B3 applied, the whole algebra collapses to:

```python
ArrayLayout = (
    LinearLayout | RectangularLayout | ImageLayout
    | VectorLayout | InstancedLayout)
```

**Recommendation.** Flat union plus one documented level order enforced in
`validate` and asserted in factories.

### B5. Smaller redundancies

- **`PhysicalStorageDomain` is unneeded and undefined.** It appears in
  `LinearLayout.size` and `PhysicalStorageObject.element_extent`, and is listed as
  remaining decision #1. A `LinearLayout` is one-dimensional by construction, so a
  multidimensional domain cannot describe its extent. The only multidimensional case
  is `ImageLayout`, which already has `physical_shape`. Delete the type; make
  `LinearLayout.size` an `ArithmeticExpression` and let `ImageLayout` carry image
  dimensions. Remaining decision #1 disappears.
- **`InstanceScope` is a function of `StorageKind` in v1.** `GLOBAL_BUFFER`/
  `CONSTANT_BUFFER`/`IMAGE` → `GLOBAL`, `LOCAL_MEMORY` → `WORKGROUP`,
  `PRIVATE_MEMORY` → `WORK_ITEM`. Storing both on `PhysicalStorageObject` invites
  the two to disagree. Make one derived (a property), and introduce the second axis
  when a case actually needs it (for example a per-workgroup global buffer).
- **`StorageKind` itself is a function of the layout tree.** Unwrapped → buffer,
  `ImageLayout` terminal → image, under a scope wrapper → local/private. Present it
  as a derived summary, not as independent state that must be kept consistent.
- **`map_expr` and `map_parameters`** are two protocol methods for one operation
  applied to two representations. A single `map_expressions(mapper)` that dispatches
  internally is simpler and removes the question of which one a caller wants.
- **`epoch_key` on `StorageReference`** is, at any access site, always the current
  epoch value (that is the stated invariant). It therefore contributes nothing to
  code generation. If liveness is its only consumer, it belongs in the liveness
  query, not in every lowered access.

---

## C. Realizability

### C1. There is no single ISL map `L` — and the common case is the non-affine one

The design repeatedly speaks of "the combined layout map" as one object that can be
composed, ranged, and used in collision queries. It cannot be, and the reason is not
exotic.

Loopy's ordinary strided array argument has symbolic strides:
`make_temporaries_for_offsets_and_strides` replaces `lp.auto` strides with `ValueArg`s,
and the address expression becomes `a_stride_0 * i + a_stride_1 * j`. A **parameter
times an index is not quasi-affine**, so a plain strided argument — the single most
common array in loopy — has a terminal address that ISL cannot represent at all.

This is why the design is right to make `LinearLayout.expr` Pymbolic while lanes,
selectors, and scope mappings are `PwAff`. But the consequence is not drawn:

- **Allocation.** "For each object, range the terminal physical coordinate over one
  allocation fiber" is not computable for a symbolic-stride layout. The design is
  saved by `RectangularLayout` deriving its extent structurally and by generic layouts
  requiring an explicit contract — but the general step as written is misleading.
  State that ranging applies only to layouts whose terminal is quasi-affine, and that
  everything else derives structurally or declares.
- **Race analysis.** Phase 9's headline is "replace syntactic write-race checks with a
  two-copy collision query ... require equal storage object and instance identity ...
  overlapping physical footprints". For a symbolic-stride argument the physical
  coordinate is not an ISL expression, so this query is unavailable exactly where it
  is most needed.

The design does provide the answer — "For scalar accesses to one array under the
layout-provider injectivity contract, equality of universal logical indices remains a
sound optimization" — but frames it as an *optimization* over the physical criterion.
It is the other way around:

> **Logical-index collision under the injectivity contract is the primary criterion.**
> Physical-footprint composition is the fallback, used only where logical index spaces
> differ (base-storage aliases, callable views with different namespaces) or where the
> footprint is not a point (vector lanes, image texels, atomic granularity).

This inversion matters for the plan: it makes Phase 9 mostly a matter of replacing
the syntactic iname-occurrence test with a two-copy ISL query on *logical* indices —
which is tractable, is the stated correctness goal (`i % 2`, `i-i`), and is what
`AGENTS.md` asks for — while the physical-footprint machinery is a narrower addition.
Phase 9 as written reads as if the physical path is the main path, which risks the
phase stalling on unrepresentable coordinates.

### C2. Gap: nothing says how an `auto` temporary gets its *layout*

The design states that `auto` infers the **exact polyhedral union** of accesses, and
that bounding-box inference is removed entirely ("Bounding-box shape inference is not
provided", Phase 2 task 6: "Remove all bounding-box shape inference and its options").

But allocation now comes from the *layout*, and an inferred temporary has no
user-supplied layout. Something must choose one, and for a triangular or
union-shaped access set the only practical automatic choice is a rectangular layout
over the bounding box of that union.

So the bounding box does not disappear; it moves from shape inference to layout
inference. That is the correct outcome — the design's own objection ("it changes
logical validity, not merely allocation") applies to shapes, not layouts, and a
bounding-box *layout* changes only allocation — but the design never says it.

**Recommendation.** Add an explicit rule: an `auto` temporary gets
`shape = exact access union` and, absent a user layout,
`layout = make_c_layout` over the bounding box of that union, with origins taken from
the box's lower bounds. Note that this is sound (it over-allocates, never
under-allocates, and does not widen the logical validity set) and that it is the
one place a bounding box legitimately appears. Otherwise Phase 2 and Phase 5 are
mutually unsatisfiable.

### C3. Shape equality and hashing need a canonical form, not `.equals()`

The design asks for two things that are in tension:

- "Set equality must be semantic after named-space alignment";
- "Persistent hashes should use a stable normalized representation".

If `__eq__` is `.equals()` (semantic) and `__hash__` is over a normalized form, then
any pair of semantically-equal sets that normalize differently gives `a == b` with
`hash(a) != hash(b)` — silent dict/cache corruption, in a codebase that hashes
kernels constantly.

The good news, checked against `namedisl` in this tree: ISL's normalization is
stronger than I expected. After `coalesce().detect_equalities().remove_redundancies()`,
all of the following normalize to identical strings:
`0<=i<n` vs `0<=i and i<=n-1`; `0<=i<5 or 5<=i<10` vs `0<=i<10`;
`exists a: i=2a` vs `i mod 2 = 0`; `0<=i<n and 0<=j<n and i<=j` vs `0<=i<=j<n`.
The one difference found is **parameter order**: `[n, m] -> ...` and `[m, n] -> ...`
print differently for equal sets.

**Recommendation.** Define the canonical form as
`coalesce → detect_equalities → remove_redundancies` with parameters sorted by name,
and define `__eq__` as equality *of that form*, not `.equals()`. Then eq/hash
consistency holds by construction and no canonicality guarantee from ISL is required.
Expose `.is_semantically_equal(other)` separately for the places that genuinely need
`.equals()` (bounds containment, descriptor matching). Add a fuzz test comparing the
two relations; where they diverge you get a conservative cache miss rather than a
correctness bug. Phase 0 task 3 should be worded this way, because "prototype
semantic equality and a stable hash" as currently written asks for something that may
not exist.

### C4. The plan never schedules `loopy/transform/`

`ARRAY-DESIGN.md` lists "Existing transformations are not redesigned here" as a
non-goal, and `ARRAY-IMPL.md` says it "does not attempt to migrate all existing
transformations". But the plan changes `.shape` from a tuple to a `namedisl.Set`,
and removes `dim_tags`, `offset`, `storage_shape`, and `base_indices` as stored
state. Every transform that touches those breaks on day one.

Measured in tree: **94 references to `.shape` or `dim_tags` across 11 transform
modules** — `padding.py` (27), `data.py` (19), `privatize.py` (11), `concatenate.py`
(7), `batch.py` (6), `pack_and_unpack_args.py` (6), `diff.py` (6), `precompute.py` (5),
`save.py` (3), `callable.py` (3), `buffer.py` (1). Several construct shape tuples
element-wise (`padding.py` reorders axes and strides; `batch.py` prepends a batch
axis; `concatenate.py` sums extents).

No phase lists any file under `loopy/transform/` in its *Primary files*, yet Phase 1's
exit criterion is "resolved array shapes are named sets throughout the modified core
paths" and Phase 4 removes stored dim tags outright.

**Recommendation.** Either:

1. **Schedule it.** Add a transform-migration phase between Phase 4 and Phase 5, with
   a per-module inventory. `padding.py` and `data.py` alone are substantial: padding
   is *entirely* about physical layout and is the transform most affected by the
   layout model. This is the honest option and probably a phase of its own.
2. **Or take the `.index_set` fallback as the plan, not the fallback.** Add
   `.index_set` as canonical, keep `.shape` returning `rectangular_shape()` with a
   deprecation warning, and let transforms migrate incrementally. The design currently
   calls this "a fallback, not the preferred design" and defers the choice to Phase 0
   go/no-go criteria — but the criterion "all in-tree core consumers can migrate
   atomically" is already answerable: with 11 transform modules and 94 sites, they
   cannot. Resolving remaining-decision #5 now, in favour of `.index_set`, would
   unblock the sequencing.

### C5. Smaller realizability notes

- **Vector lane determination may regress.** The design requires composing
  `lane_expr` (a `PwAff`) with the exact instruction-to-logical-index map, and calls
  failure to build that map "an actionable error, not a reason to approximate". Today,
  a scalar vector-axis index is resolved by `eval_expr_assert_integer_constant`
  (`loopy/kernel/array.py:1270`), which evaluates under the *unrolled* codegen context.
  An index that is a compile-time constant after unrolling but not quasi-affine as a
  function of the domain is accepted today and would be rejected under the new rule.
  Rare, but it is a silent compatibility break. Keep the constant-evaluation path as a
  fallback when the index is already compile-time constant in the current context.
- **Runtime parameter inference is more realizable than the design suggests.** The
  "layout-contributed equations" refactor maps almost exactly onto the existing
  `_ArgFindingEquation` machinery in `loopy/target/execution.py:185-320`, which already
  has `order` (for dependency staging), `based_on_names` (for "don't use two facts from
  one array"), and `solve_affine_equations_for`. Worth naming in Phase 5 task 7 so the
  scope is understood as a refactor rather than a rewrite.
- **`AccessFootprint` is underspecified.** It is declared as a bare `class ...: ...`
  with a comment, yet Phase 9 correctness depends on its overlap relation. It needs at
  minimum: scalar element, lane subset of a vector, whole vector, image texel, and
  target atomic granularity — plus a defined, conservative-by-default `overlaps`
  operation.
- **Eleven phases with a coexisting legacy path.** From Phase 4 (dim tags no longer
  stored) to Phase 10 (automatic universalization enabled), code generation must work
  from layouts while arrays still carry old address spaces, and the legacy path must
  run off the computed `.dim_tags` compatibility view. That view is defined to exist
  only "for layouts exactly representable by old tags", so any kernel using a new
  layout feature has no legacy path. That is acceptable, but the plan should say so:
  new layout features are available only on the new path, and the compatibility suite
  must not require them on the old one.

---

## D. What is right and should not change

For balance, the following are well judged and should survive any rescoping:

- Shape as a named index set, with `None` banned and `auto` as a construction-time
  request only. Commit `acf3fde6` already started this.
- Bounds checking as set containment rather than per-axis extents — this fixes a real
  class of bugs (points inside the hull but outside the set).
- Allocation derived from layout rather than from `product(shape) * itemsize`.
- Injectivity as a **provider contract** with structural `validate` explicitly not
  attempting a collision proof. The reasoning given — `floor(i/4)` and `i mod 4` are
  each noninjective while their pair is injective, so per-component checks prove
  nothing — is exactly right, and the alternative (a general two-copy solver in the
  validation path) would be both slow and incomplete.
- Refusing to synthesize read-modify-write for partial writes (keep the principle even
  if the image case goes away per B1).
- Keeping construction policy in `make_*_layout` functions with trivial dataclass
  record constructors, per `AGENTS.md`.
- Separating ownership and liveness from storage kind. Today's conflation of "global"
  with "host-allocated and passed between subkernels" is a real source of confusion,
  and the design untangles it cleanly.

---

## E. Suggested rescoped v1

Applying A1–A3 and B1–B5, the model becomes:

```text
shape   : namedisl.Set                       (logical index set, no instance dims)
layout  : ArrayLayout                        (flat union, 5 classes)
          LinearLayout | RectangularLayout | ImageLayout
          | VectorLayout | InstancedLayout
sep     : construction-time option, expanded in preprocessing (as today)
storage : StorageKind / InstanceScope derived from the layout tree
phase   : universalization recorded in KernelState, not per array
```

with the primary race criterion being logical-index collision under the injectivity
contract, and physical-footprint composition reserved for aliases and non-point
footprints.

That retains every stated **goal** — nonrectangular shapes, explicit allocation
instances, layouts replacing dim tags, compositional vector representation, one
representation for all consumers, preserved rectangular behavior, retained legacy
entry points — while removing the universalization access-rewriting pass, the
`HardwareAxisMapping` machinery, the image-vector feature, the swizzle target hook,
the separate-layout algebra, the generic type parameters, and two of the five
remaining prototype decisions.

The single largest remaining risk is not in the design at all: it is C4, the 11
transform modules and 94 call sites that no phase currently owns.
