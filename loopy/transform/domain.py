from __future__ import annotations


__copyright__ = "Copyright (C) 2023 Kaushik Kulkarni"

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

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: decouple_domain
"""


from typing import TYPE_CHECKING

import islpy as isl

from loopy.diagnostic import LoopyError
from loopy.translation_unit import for_each_kernel


if TYPE_CHECKING:
    from collections.abc import Collection

    from loopy.kernel import LoopKernel


@for_each_kernel
def decouple_domain(kernel: LoopKernel,
                    inames: Collection[str],
                    parent_inames: Collection[str]) -> LoopKernel:
    r"""
    Returns a copy of *kernel* with altered domains. The home domain of
    *inames* i.e. :math:`\mathcal{D}^{\text{home}}({\text{inames}})` is
    replaced with two domains :math:`\mathcal{D}_1` and :math:`\mathcal{D}_2`.
    :math:`\mathcal{D}_1` is the domain with dimensions corresponding to *inames*
    projected out and :math:`\mathcal{D}_2` is the domain with all the dimensions
    other than the ones corresponding to *inames* projected out.

    :arg inames: The inamaes to be decouple from their home domain.
    :arg parent_inames: Inames in :math:`\mathcal{D}^{\text{home}}({\text{inames}})`
        that will be used as additional parametric dimensions during the
        construction of :math:`\mathcal{D}_1`.

    .. note::

        - An error is raised if all the *inames* do not correspond to the same home
          domain of *kernel*.
        - It is the caller's responsibility to ensure that :math:`\mathcal{D}_1
          \cup \mathcal{D}_2 = \mathcal{D}^{\text{home}}({\text{inames}})`. If this
          criterion is violated this transformation would violate dependencies.
    """

    # {{{ sanity checks

    if not inames:
        raise LoopyError("No inames were provided to decouple into"
                         " a different domain.")
    if frozenset(parent_inames) & frozenset(inames):
        raise LoopyError("Inames cannot be appear in `inames` and `parent_inames`.")

    # }}}

    hdi = kernel.get_home_domain_index(next(iter(inames)))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != hdi:
            raise LoopyError("inames are not a part of the same home domain.")

    all_dims = frozenset(kernel.domains[hdi].get_var_dict())
    for parent_iname in parent_inames:
        if parent_iname not in all_dims:
            raise LoopyError(f"Parent iname '{parent_iname}' not a part of the"
                             f" corresponding home domain '{kernel.domains[hdi]}'.")

    dom1 = kernel.domains[hdi]
    dom2 = kernel.domains[hdi]

    for iname in sorted(all_dims):
        if iname in inames:
            dt, pos = dom1.get_var_dict()[iname]
            dom1 = dom1.project_out(dt, pos, 1)
        elif iname in parent_inames:
            dt, pos = dom2.get_var_dict()[iname]
            if dt != isl.dim_type.param:
                n_params = dom2.dim(isl.dim_type.param)
                dom2 = dom2.move_dims(isl.dim_type.param, n_params, dt, pos, 1)
        else:
            dt, pos = dom2.get_var_dict()[iname]
            dom2 = dom2.project_out(dt, pos, 1)

    new_domains = list(kernel.domains)
    new_domains[hdi] = dom1
    new_domains.append(dom2)
    kernel = kernel.copy(domains=new_domains)
    return kernel

# vim: fdm=marker
