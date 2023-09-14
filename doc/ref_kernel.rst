.. currentmodule:: loopy

Reference: Loopy's Model of a Kernel
====================================

What Types of Computation can a Loopy Program Express?
------------------------------------------------------

Loopy programs consist of an a-priori unordered set of statements, operating
on :math:`n`-dimensional array variables.

Arrays consist of "plain old data" and structures thereof, as describable
by a :class:`numpy.dtype`.  The n-dimensional shape of these arrays is
given by a tuple of expressions at most affine in parameters that are
fixed for the duration of program execution.
Each array variable in the program is either an argument or a temporary
variable.  A temporary variable is only live within the program, while
argument variables are accessible outside the program and constitute the
program's inputs and outputs.

A statement (still called 'instruction' in some places, cf.
:class:`loopy.InstructionBase`) encodes an assignment to an entry of an array.
The right-hand side of an assignment consists of an expression that may
consist of arithmetic operations and calls to functions.
If the outermost operation of the RHS expression is a function call,
the RHS value may be a tuple, and multiple (still scalar) arrays appear
as LHS values. (This is the only sense in which tuple types are supported.)
Each statement is parametrized by zero or more loop variables ("inames").
A statement is executed once for each integer point defined by the domain
forest for the iname tuple given for that statement
(:attr:`loopy.InstructionBase.within_inames`). Each execution of a
statement (with specific values of the inames) is called a *statement
instance*.  Dependencies between these instances as well as instances of
other statements are encoded in the program representation and specify permissible
execution orderings.  (The semantics of the dependencies are `being
sharpened <https://github.com/inducer/loopy/pull/168>`__.) Assignments
(comprising the evaluation of the RHS and the assignment to the LHS) may
be specified to be atomic.

The basic building blocks of the domain forest are sets given as
conjunctions of equalities and inequalities of quasi-affine expressions on
integer tuples, called domains, and represented as instances of
:class:`islpy.BasicSet`. The entries of each integer tuple are
either *parameters* or *inames*. Each domain may optionally have a *parent
domain*. Parameters of parent-less domains are given by value arguments
supplied to the program that will remain unchanged during program
execution. Parameters of domains with parents may be

- run-time-constant value arguments to the program, or
- inames from parent domains, or
- scalar, integer temporary variables that are written by statements
  with iteration domains controlled by a parent domain.

For each tuple of concrete parameter values, the set of iname tuples must be
finite. Each iname is defined by exactly one domain.

For a tuple of inames, the domain forest defines an iteration domain
by finding all the domains defining the inames involved, along with their
parent domains. The resulting tree of domains may contain multiple roots,
but no branches. The iteration domain is then constructed by intersecting
these domains and constructing the projection of that set onto the space
given by the required iname tuple. Observe that, via the parent-child
domain mechanism, imperfectly-nested and data-dependent loops become
expressible.

The set of functions callable from the language is predefined by the system.
Additional functions may be defined by the user by registering them. It is
not currently possible to define functions from within Loopy, however work
is progressing on permitting this. Even once this is allowed, recursion
will not be permitted.

.. _domain-tree:

Loop Domain Forest
------------------

.. {{{

Example::

    { [i]: 0<=i<n }

A kernel's iteration domain is given by a list of :class:`islpy.BasicSet`
instances (which parametrically represent multi-dimensional sets of
tuples of integers).  They define the integer values of the loop variables
for which instructions (see below) will be executed.
It is written in :ref:`isl-syntax`.  :mod:`loopy` calls the loop variables
*inames*. In this case, *i* is the sole iname. The loop
domain is given as a conjunction of affine equality
and inequality constraints. Integer divisibility constraints (resulting
in strides) are also allowed. In the absence of divisibility
constraints, the loop domain is convex.

Note that *n* in the example is not an iname. It is a
:ref:`domain-parameters` that is passed to the kernel by the user.

To accommodate some data-dependent control flow, there is not actually
a single loop domain, but rather a *forest of loop domains* (a collection
of trees) allowing more deeply nested domains to depend on inames
introduced by domains closer to the root.

Here is an example::

    { [l] : 0 <= l <= 2 }
      { [i] : start <= i < end }
      { [j] : start <= j < end }

The i and j domains are "children" of the l domain (visible from indentation).
This is also how :mod:`loopy` prints the domain forest, to make the parent/child
relationship visible.  In the example, the parameters start/end might be read
inside of the 'l' loop.

The idea is that domains form a forest (a collection of trees), and a
"sub-forest" is extracted that covers all the inames for each
instruction. Each individual sub-tree is then checked for branching,
which is ill-formed. It is declared ill-formed because intersecting, in
the above case, the l, i, and j domains could result in restrictions from the
i domain affecting the j domain by way of how i affects l--which would
be counterintuitive to say the least.)

.. _inames:

Inames
^^^^^^

Loops are (by default) entered exactly once. This is necessary to preserve
dependency semantics--otherwise e.g. a fetch could happen inside one loop nest,
and then the instruction using that fetch could be inside a wholly different
loop nest.

.. _isl-syntax:

ISL syntax
^^^^^^^^^^

The general syntax of an ISL set is the following::

    {[VARIABLES]: CONDITIONS}

``VARIABLES`` is a simple list of identifiers representing loop indices,
or, as loopy calls them, inames. Example::

    {[i, j, k]: CONDITIONS}

The following constructs are supported for ``CONDITIONS``:

* Simple conditions: ``i <= 15``, ``i>0``

* Conjunctions: ``i > 0 and i <= 15``

* Two-sided conditions: ``0 < i <= 15`` (equivalent to the previous
  example)

* Identical conditions on multiple variables:
  ``0 < i,j <= 15``

* Equality constraints: ``i = j*3`` (**Note:** ``=``, not ``==``.)

* Modulo: ``i mod 3 = 0``

* Existential quantifiers: ``(exists l: i = 3*l)`` (equivalent to the
  previous example)

Examples of constructs that are **not** allowed:

* Multiplication by non-constants: ``j*k``

* Disjunction: ``(i=1) or (i=5)``
  (**Note:** This may be added in a future version of loopy.
  For now, loop domains have to be convex.)

.. _domain-parameters:

Domain parameters
^^^^^^^^^^^^^^^^^

Domain parameters are identifiers being used in loop domains that are not
*inames*, i.e. they do not define loop variables. In the following domain
specification, *n* is a domain parameter::

    {[i,j]: 0 <= i,j < n}

Values of domain parameters arise from

* being passed to the kernel as :ref:`arguments`

* being assigned to :ref:`temporaries` to feed into domains
  lower in the :ref:`domain-tree`.

.. _iname-tags:

Iname Implementation Tags
^^^^^^^^^^^^^^^^^^^^^^^^^

=============================== ====================================================
Tag                             Meaning
=============================== ====================================================
``None`` | ``"for"``            Sequential loop
``"ord"``                       Forced-order sequential loop
``"l.N"``                       Local (intra-group) axis N ("local")
``"g.N"``                       Group-number axis N ("group")
``"unr"``                       Unroll
``"unr_hint"``                  Unroll using compiler directives
``"unr_hint.N"``                Unroll at most N times using compiler directives
``"ilp"`` | ``"ilp.unr"``       Unroll using instruction-level parallelism
``"ilp.seq"``                   Realize parallel iname as innermost loop
``"like.INAME"``                Can be used when tagging inames to tag like another
``"unused.g"`` | ``"unused.l"`` Can be to tag as the next unused group/local axis
=============================== ====================================================

(Throughout this table, `N` must be replaced by an actual, zero-based number.)

"ILP" does three things:

* Restricts loops to be innermost
* Duplicates reduction storage for any reductions nested around ILP usage
* Causes a loop (unrolled or not) to be opened/generated for each
  involved instruction

.. }}}

Identifiers
-----------

Reserved Identifiers
^^^^^^^^^^^^^^^^^^^^

The identifier prefix ``_lp_`` is reserved for internal usage; when creating
*inames*, *argument names*, *temporary variable names*, *substitution rule
names*, *instruction IDs*, and other identifiers, users should *not* use names
beginning with ``_lp_``.  This prefix is used for identifiers created
internally when operating on Loopy's kernel IR. For Loopy developers, further
information on name prefixes used within submodules is below.

Identifier Registry
^^^^^^^^^^^^^^^^^^^

Functionality in :mod:`loopy` *must* use identifiers beginning with ``_lp_`` for
all internally-created identifiers. Additionally, each name beginning with
``_lp_`` must start with one of the reserved prefixes below. New prefixes may
be registered by adding them to the table below. New prefixes may not themselves
be the prefix of an existing prefix.

**Reserved Identifier Prefixes**

======================= ==================================
Reserved Prefix         Usage (module or purpose)
======================= ==================================
``_lp_linchk_``         ``loopy.linearization.checker``
======================= ==================================

.. note::

    Existing Loopy code may not yet fully satisfy these naming requirements.
    Name changes are in progress, and prefixes will be added to this registry
    as they are created.

.. _instructions:

Instructions
------------

.. {{{

.. autoclass:: InstructionBase

.. _assignments:

Assignment objects
^^^^^^^^^^^^^^^^^^

.. autoclass:: Assignment

.. _assignment-syntax:

Textual Assignment Syntax
^^^^^^^^^^^^^^^^^^^^^^^^^

The general syntax of an instruction is a simple assignment::

    LHS[i,j,k] = EXPRESSION

Several extensions of this syntax are defined, as discussed below.  They
may be combined freely.

You can also use an instruction to declare a new temporary variable. (See
:ref:`temporaries`.) See :ref:`types` for what types are acceptable. If the
``LHS`` has a subscript, bounds on the indices are inferred (which must be
constants at the time of kernel creation) and the declared temporary is
created as an array. Instructions declaring temporaries have the following
form::

    <temp_var_type> LHS[i,j,k] = EXPRESSION

You can also create a temporary and ask loopy to determine its type
automatically. This uses the following syntax::

    <> LHS[i,j,k] = EXPRESSION

Lastly, each instruction may optionally have a number of attributes
specified, using the following format::

    LHS[i,j,k] = EXPRESSION {attr1,attr2=value1:value2}

These are usually key-value pairs. The following attributes are recognized:

* ``id=value`` sets the instruction's identifier to ``value``. ``value``
  must be unique within the kernel. This identifier is used to refer to the
  instruction after it has been created, such as from ``dep`` attributes
  (see below) or from :mod:`context matches <loopy.match>`.

* ``id_prefix=value`` also sets the instruction's identifier, however
  uniqueness is ensured by loopy itself, by appending further components
  (often numbers) to the given ``id_prefix``.

* ``inames=i:j:k`` forces the instruction to reside within the loops over
  :ref:`inames` ``i``, ``j`` and ``k`` (and only those).

  .. note::

      The default for the inames that the instruction depends on is
      the inames used in the instruction itself plus the common
      subset of inames shared by writers of all variables read by the
      instruction.

      You can add a plus sign ("``+``") to the front of this option
      value to indicate that you would like the inames you specify here
      to be in addition to the ones found by the heuristic described above.

* ``dup=i:j->j_new:k->k_new`` makes a copy of the inames ``i``, ``j``, and
  ``k``, with all the same domain constraints as the original inames.
  A new name of the copy of ``i`` will be automatically chosen, whereas
  the new name of ``j`` will be ``j_new``, and the new name of ``k`` will
  be ``k_new``.

  This is a shortcut for calling :func:`loopy.duplicate_inames` later
  (once the kernel is created).

* ``dep=id1:id2`` creates a dependency of this instruction on the
  instructions with identifiers ``id1`` and ``id2``. The meaning of this
  dependency is that the code generated for this instruction is required to
  appear textually after all of these dependees' generated code.

  Identifiers here are allowed to be wildcards as defined by the Python
  function :func:`fnmatch.fnmatchcase`. This is helpful in conjunction with
  ``id_prefix``.

  .. note::

      Since specifying all possible dependencies is cumbersome and
      error-prone, :mod:`loopy` employs a heuristic to automatically find
      dependencies. Specifically, :mod:`loopy` will automatically add
      a dependency to an instruction reading a variable if there is
      exactly one instruction writing that variable. ("Variable" here may
      mean either temporary variable or kernel argument.)

      If each variable in a kernel is only written once, then this
      heuristic should be able to compute all required dependencies.

      Conversely, if a variable is written by two different instructions,
      all ordering around that variable needs to be specified explicitly.
      It is recommended to use :func:`get_dot_dependency_graph` to
      visualize the dependency graph of possible orderings.

      You may use a leading asterisk ("``*``") to turn off the single-writer
      heuristic and indicate that the specified list of dependencies is
      exhaustive.

* ``dep_query=...`` provides an alternative way of specifying instruction
  dependencies. The given string is parsed as a match expression object by
  :func:`loopy.match.parse_match`. Upon kernel generation, this match
  expression is used to match instructions in the kernel and add them as
  dependencies.

* ``nosync=id1:id2`` prescribes that no barrier synchronization is necessary
  for the instructions with identifiers ``id1`` and ``id2``, even if a
  dependency chain exists and variables are accessed in an apparently racy
  way.

  Identifiers here are allowed to be wildcards as defined by the Python
  function :func:`fnmatch.fnmatchcase`. This is helpful in conjunction with
  ``id_prefix``.

  Identifiers (including wildcards) accept an optional `@scope` suffix,
  which prescribes that no synchronization at level `scope` is needed.
  This does not preclude barriers at levels different from `scope`.
  Allowable `scope` values are:

  * `local`
  * `global`
  * `any`

  As an example, ``nosync=id1@local:id2@global`` prescribes that no local
  synchronization is needed with instruction ``id1`` and no global
  synchronization is needed with instruction ``id2``.

  ``nosync=id1@any`` has the same effect as ``nosync=id1``.

* ``nosync_query=...`` provides an alternative way of specifying ``nosync``,
  just like ``dep_query`` and ``dep``. As with ``nosync``, ``nosync_query``
  accepts an optional `@scope` suffix.

* ``priority=integer`` sets the instructions priority to the value
  ``integer``. Instructions with higher priority will be scheduled sooner,
  if possible. Note that the scheduler may still schedule a lower-priority
  instruction ahead of a higher-priority one if loop orders or dependencies
  require it.

* ``if=variable1:variable2`` Only execute this instruction if all condition
  variables (which must be scalar variables) evaluate to ``true`` (as
  defined by C).

* ``tags=tag1:tag2`` Apply tags to this instruction that can then be used
  for :ref:`context-matching`.

* ``groups=group1:group2`` Make this instruction part of the given
  instruction groups. See :class:`InstructionBase.groups`.

* ``conflicts_grp=group1:group2`` Make this instruction conflict with the
  given instruction groups. See
  :class:`InstructionBase.conflicts_with_groups`.

* ``atomic`` The update embodied by the assignment is carried out
  atomically. See :attr:`Assignment.atomicity` for precise semantics.

.. _expression-syntax:

Expressions
^^^^^^^^^^^

Loopy's expressions are a slight superset of the expressions supported by
:mod:`pymbolic`.

* ``if(cond, then, else_)``

* ``a[[ 8*i + j ]]``: Linear subscripts.
  See :class:`loopy.symbolic.LinearSubscript`.

* ``reductions``
  See :class:`loopy.symbolic.Reduction`.

    * ``reduce`` vs ``simul_reduce``

* complex-valued arithmetic

* tagging of array access and substitution rule use ("$")
  See :class:`loopy.symbolic.TaggedVariable`.

* ``indexof``, ``indexof_vec``
* ``cast(type, value)``: No parse syntax currently.
  See :class:`loopy.symbolic.TypeCast`.

* If constants in expressions are subclasses of :class:`numpy.generic`,
  generated code will contain literals of exactly that type, making them
  *explicitly typed*. Constants given as Python types such as :class:`int`,
  :class:`float` or :class:`complex` are called *implicitly* typed and
  adapt to the type of the expected result.

TODO: Functions
TODO: Reductions

Function Call Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: loopy
.. autoclass:: CallInstruction

C Block Instructions
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CInstruction

Atomic Operations
^^^^^^^^^^^^^^^^^

.. autoclass:: MemoryOrdering

.. autoclass:: MemoryScope

.. autoclass:: VarAtomicity

.. autoclass:: OrderedAtomic

.. autoclass:: AtomicInit

.. autoclass:: AtomicUpdate

No-Op Instruction
^^^^^^^^^^^^^^^^^

.. autoclass:: NoOpInstruction

Barrier Instructions
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BarrierInstruction

Instruction Tags
^^^^^^^^^^^^^^^^

.. autoclass:: LegacyStringInstructionTag
.. autoclass:: UseStreamingStoreTag

.. }}}

Data: Arguments and Temporaries
-------------------------------

.. {{{

Kernels operate on two types of data: 'arguments' carrying data into and out of a kernel,
and temporaries with lifetimes tied to the runtime of the kernel.

.. _arguments:

Arguments
^^^^^^^^^

.. autoclass:: KernelArgument

.. autoclass:: ValueArg

.. autoclass:: ArrayArg

.. autoclass:: ConstantArg

.. autoclass:: ImageArg

.. _temporaries:

Temporary Variables
^^^^^^^^^^^^^^^^^^^

Temporary variables model OpenCL's ``private`` and ``local`` address spaces. Both
have the lifetime of a kernel invocation.

.. autoclass:: AddressSpace

.. autoclass:: TemporaryVariable

.. _types:

Specifying Types
^^^^^^^^^^^^^^^^

:mod:`loopy` uses the same type system as :mod:`numpy`. (See
:class:`numpy.dtype`) It also uses :mod:`pyopencl` for a registry of
user-defined types and their C equivalents. See :func:`pyopencl.tools.get_or_register_dtype`
and related functions.

For a string representation of types, all numpy types (e.g. ``float32`` etc.)
are accepted, in addition to what is registered in :mod:`pyopencl`.

.. _data-dim-tags:

Data Axis Tags
^^^^^^^^^^^^^^

Data axis tags specify how a multi-dimensional array (which is loopy's
main way of storing data) is represented in (linear, 1D) computer
memory. This storage format is given as a number of "tags", as listed
in the table below. Each axis of an array has a tag corresponding to it.
In the user interface, array dim tags are specified as a tuple of these
tags or a comma-separated string containing them, such as the following::

    c,vec,sep,c

The interpretation of these tags is order-dependent, they are read
from left to right.

===================== ====================================================
Tag                   Meaning
===================== ====================================================
``c``                 Nest current axis around the ones that follow
``f``                 Nest current axis inside the ones that follow
``N0`` ... ``N9``     Specify an explicit nesting level for this axis
``stride:EXPR``       A fixed stride
``sep``               Implement this axis by mapping to separate arrays
``vec``               Implement this axis as entries in a vector
===================== ====================================================

``sep`` and ``vec`` obviously require the number of entries
in the array along their respective axis to be known at code
generation time.

When the above speaks about 'nesting levels', this means that axes
"nested inside" others are "faster-moving" when viewed from linear
memory.

In addition, each tag may be followed by a question mark (``?``),
which indicates that if there are more dimension tags specified
than array axes present, that this axis should be omitted. Axes
with question marks are omitted in a left-first manner until the correct
number of dimension tags is achieved.

Some examples follow, all of which use a three-dimensional array of shape
*(3, M, 4)*. For simplicity, we assume that array entries have size one.

*   ``c,c,c``: The axes will have strides *(M*4, 4, 1)*,
    leading to a C-like / row-major layout.

*   ``f,f,f``: The axes will have strides *(1, 3, 3*M)*,
    leading to a Fortran-like / row-major layout.

*   ``sep,c,c``: The array will be mapped to three arrays of
    shape *(M, 4)*, each with strides *(4, 1)*.

*   ``c,c,vec``: The array will be mapped to an array of
    ``float4`` vectors, with (``float4``-based) strides of
    *(M, 1)*.

*   ``N1,N0,N2``: The axes will have strides *(M, 1, 3*M)*.

.. }}}

.. _substitution-rule:

Substitution Rules
------------------

.. {{{

Substitution Rule Objects
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SubstitutionRule

.. _subst-rule-syntax:

Textual Syntax for Substitution Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax of a substitution rule::

    rule_name(arg1, arg2) := EXPRESSION

.. }}}

Kernel Options
--------------

.. autoclass:: Options

.. _targets:

Targets
-------

.. automodule:: loopy.target

.. currentmodule:: loopy

Helper values
-------------

.. {{{

.. autoclass:: auto

.. autoclass:: UniqueName

.. autoclass:: Optional

.. }}}

Libraries: Extending and Interfacing with External Functionality
----------------------------------------------------------------

.. _symbols:

Symbols
^^^^^^^

.. _functions:

Functions
^^^^^^^^^

.. autoclass:: PreambleInfo

.. autoclass:: CallMangleInfo

.. _reductions:

Reductions
^^^^^^^^^^


The Kernel Object
-----------------

Do not create :class:`LoopKernel` objects directly. Instead, refer to
:ref:`creating-kernels`.

.. autoclass:: LoopKernel

.. autoclass:: KernelState
    :members:
    :undoc-members:

Implementation Details: The Base Array
--------------------------------------

All array-like data in :mod:`loopy` (such as :class:`ArrayArg` and
:class:`TemporaryVariable`) derive from single, shared base array type,
described next.

.. currentmodule:: loopy.kernel.array

.. autoclass:: ArrayBase


.. vim: tw=75:spell:fdm=marker
