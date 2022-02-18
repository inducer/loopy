"""Loop nest build top-level control/hoisting."""


__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def synthesize_idis_for_extra_args(kernel, sched_item):
    """
    :arg kernel: An instance of :class:`loopy.LoopKernel`.
    :arg sched_item: An instance of :class:`loopy.schedule.tree.Function`.
    :returns: A list of :class:`loopy.codegen.ImplementedDataInfo`
    """
    from loopy.codegen import ImplementedDataInfo
    from loopy.kernel.data import InameArg, AddressSpace
    from loopy.schedule.tree import Function

    assert isinstance(sched_item, Function)

    idis = []

    for arg in sched_item.extra_args:
        temporary = kernel.temporary_variables[arg]
        assert temporary.address_space == AddressSpace.GLOBAL
        idis.extend(
            temporary.decl_info(
                kernel.target,
                index_dtype=kernel.index_dtype))

    for iname in sched_item.extra_inames:
        idis.append(
            ImplementedDataInfo(
                target=kernel.target,
                name=iname,
                dtype=kernel.index_dtype,
                arg_class=InameArg,
                is_written=False))

    return idis

# vim: foldmethod=marker
