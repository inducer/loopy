Reference: Documentation for Internal API
=========================================

Targets
-------

See also :ref:`targets`.

.. automodule:: loopy.target.c

Symbolic
--------

See also :ref:`expression-syntax`.

.. automodule:: loopy.symbolic

Types
-----

DTypes of variables in a :class:`loopy.LoopKernel` must be picklable, so in
the codegen pipeline user-provided types are converted to
:class:`loopy.types.LoopyType`.

.. automodule:: loopy.types

Type inference
^^^^^^^^^^^^^^

.. automodule:: loopy.type_inference

Codegen
-------

.. automodule:: loopy.codegen

Reduction Operation
-------------------

.. automodule:: loopy.library.reduction

Iname Tags
----------

.. automodule:: loopy.kernel.data

Array
-----

.. automodule:: loopy.kernel.array

Checks
------

.. automodule:: loopy.check

Schedule
--------

.. automodule:: loopy.schedule
.. automodule:: loopy.schedule.tools
.. automodule:: loopy.schedule.tree

References
----------

Mostly things that Sphinx (our documentation tool) should resolve but won't.

.. class:: constantdict

    See :class:`constantdict.constantdict`.

.. class:: DTypeLike

    See :data:`numpy.typing.DTypeLike`.

.. currentmodule:: p

.. class:: Call

    See :class:`pymbolic.primitives.Call`.

.. class:: CallWithKwargs

    See :class:`pymbolic.primitives.CallWithKwargs`.

.. currentmodule:: isl

.. class:: Space

    See :class:`islpy.Space`.

.. class:: Aff

    See :class:`islpy.Aff`.

.. class:: PwAff

    See :class:`islpy.PwAff`.
