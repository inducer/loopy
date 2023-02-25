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

import islpy as isl

from loopy.translation_unit import for_each_kernel
from loopy.kernel import LoopKernel
from loopy.diagnostic import LoopyError
from collections.abc import Collection


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

    .. note::

        An error is raised if all the *inames* do not correspond to the same home
        domain of *kernel*.
    """

    if not inames:
        raise LoopyError("No inames were provided to decouple into"
                         " a different domain.")

    hdi = kernel.get_home_domain_index(next(iter(inames)))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != hdi:
            raise LoopyError("inames are not a part of the same home domain.")

    for parent_iname in parent_inames:
        if parent_iname not in set(kernel.domains[hdi].get_var_dict()):
            raise LoopyError(f"Parent iname '{parent_iname}' not a part of the"
                             f" corresponding home domain '{kernel.domains[hdi]}'.")

    all_dims = frozenset(kernel.domains[hdi].get_var_dict())
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

    new_domains = kernel.domains[:]
    new_domains[hdi] = dom1
    new_domains.append(dom2)
    kernel = kernel.copy(domains=new_domains)
    return kernel
