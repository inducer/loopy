"""Library integration with Random123."""
from __future__ import annotations


__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from constantdict import constantdict
from mako.template import Template
from typing_extensions import override

from pymbolic.typing import not_none

from loopy.kernel.function_interface import ArgDescriptor, ScalarCallable


if TYPE_CHECKING:
    from collections.abc import Mapping

    from loopy.target.c import CFamilyTarget
    from loopy.translation_unit import CallablesInferenceContext
    from loopy.types import LoopyType


# {{{ rng metadata

class BaseRNGInfo(NamedTuple):
    name: str
    pyopencl_header: str
    generic_header: str
    key_width: int


class RNGInfo(NamedTuple):
    name: str
    pyopencl_header: str
    generic_header: str
    key_width: int

    width: int
    bits: int

    @property
    def full_name(self) -> str:
        return "%s%dx%d" % (self.name, not_none(self.width), not_none(self.bits))


_philox_base_info = BaseRNGInfo(
            name="philox",
            pyopencl_header="pyopencl-random123/philox.cl",
            generic_header="Random123/philox.h",
            key_width=2)

_threefry_base_info = BaseRNGInfo(
            name="threefry",
            pyopencl_header="pyopencl-random123/threefry.cl",
            generic_header="Random123/threefry.h",
            key_width=4)

RNG_VARIANTS = [
        RNGInfo(*_philox_base_info, 2, 32),
        RNGInfo(*_philox_base_info, 2, 64),
        RNGInfo(*_philox_base_info, 4, 32),
        RNGInfo(*_philox_base_info, 4, 64),

        RNGInfo(*_threefry_base_info, 2, 32),
        RNGInfo(*_threefry_base_info, 2, 64),
        RNGInfo(*_threefry_base_info, 4, 32),
        RNGInfo(*_threefry_base_info, 4, 64),
        ]

FUNC_NAMES_TO_RNG = {
        v.full_name + suffix: v
        for v in RNG_VARIANTS
        for suffix in [
            "", "_f32", "_f64",
            ]}

# }}}


# {{{ preamble

PREAMBLE_TEMPLATE = Template("""
%if is_pyopencl_target:
#include <${ rng_variant.pyopencl_header }>
%else:
#include <${ rng_variant.generic_header }>
%endif

<%
name = rng_variant.full_name
width = rng_variant.width
if rng_variant.bits == 32:
    counter_type = "uint%d" % width
    key_type = "uint%d" % rng_variant.key_width
elif rng_variant.bits == 64:
    counter_type = "ulong%d" % width
    key_type = "ulong%d" % rng_variant.key_width
else:
    assert False
%>

typedef union {
    ${ counter_type } v;
    ${ name }_ctr_t c;
} ${ name }_ctr_vec_union;


${ counter_type } ${ name }_bump(${ counter_type } ctr)
{
    if (++ctr.x == 0)
        if (++ctr.y == 0)
            ++ctr.z;
    return ctr;
}

${ counter_type } ${ name }_gen(
        ${ counter_type } ctr,
        ${ key_type } key,
        ${ counter_type } *new_ctr)
{
    ${ name }_ctr_vec_union result;
    result.c = ${ name }(
        *(${ name }_ctr_t *) &ctr,
        *(${ name }_key_t *) &key);
    *new_ctr = ${ name }_bump(ctr);
    return result.v;
}

float${ width } ${ name }_f32(
        ${ counter_type } ctr,
        ${ key_type } key,
        ${ counter_type } *new_ctr)
{
    *new_ctr = ctr;
    return
        convert_float${ width }(${ name }_gen(*new_ctr, key, new_ctr))
        * ${ repr(1./2**32) }f;
}

double${ width } ${ name }_f64(
        ${ counter_type } ctr,
        ${ key_type } key,
        ${ counter_type } *new_ctr)
{
    *new_ctr = ctr;
    %if rng_variant.bits == 32:
        return
            convert_double${ width }(${ name }_gen(*new_ctr, key, new_ctr))
            * ${ repr(1./2**32) }
            +
            convert_double${ width }(${ name }_gen(*new_ctr, key, new_ctr))
            * ${ repr(1./2**64) };

    %elif rng_variant.bits == 64:
        *new_ctr = ctr;
        return
            convert_double${ width }(${ name }_gen(*new_ctr, key, new_ctr))
            * ${ repr(1./2**64) };

    %else:
        #error Unrecognized bit width in RNG

    %endif
}

""", strict_undefined=True)

# }}}


@dataclass(frozen=True, init=False)
class Random123Callable(ScalarCallable):
    """
    Records information about for the random123 functions.
    """
    target: CFamilyTarget

    def __init__(self,
                 name: str,
                 arg_id_to_dtype: Mapping[int | str, LoopyType] | None = None,
                 arg_id_to_descr: Mapping[int | str, ArgDescriptor] | None = None,
                 name_in_target: str | None = None,
                 target: CFamilyTarget | None = None,
             ) -> None:
        super().__init__(name=name,
                         arg_id_to_dtype=arg_id_to_dtype,
                         arg_id_to_descr=arg_id_to_descr,
                         name_in_target=name_in_target)

        object.__setattr__(self, "target", not_none(target))

    @override
    def with_types(self,
                   arg_id_to_dtype: Mapping[int | str, LoopyType],
                   clbl_inf_ctx: CallablesInferenceContext,
               ) -> tuple[ScalarCallable, CallablesInferenceContext]:

        if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype:
            # the types provided aren't mature enough to specialize the
            # callable
            return (self.copy(), clbl_inf_ctx)

        name = self.name
        target = self.target

        rng_variant = FUNC_NAMES_TO_RNG[name]

        from loopy.types import NumpyType
        base_dtype = {32: np.uint32, 64: np.uint64}[rng_variant.bits]
        ctr_dtype = target.vector_dtype(NumpyType(base_dtype), rng_variant.width)
        key_dtype = target.vector_dtype(NumpyType(base_dtype), rng_variant.key_width)

        fn = rng_variant.full_name
        if name == fn:
            new_arg_id_to_dtype = {-1: ctr_dtype, -2: ctr_dtype, 0: ctr_dtype, 1:
                    key_dtype}
            return (
                    self.copy(arg_id_to_dtype=constantdict(new_arg_id_to_dtype),
                              name_in_target=fn+"_gen"),
                    clbl_inf_ctx)

        elif name == fn + "_f32":
            new_arg_id_to_dtype = {-1: target.vector_dtype(NumpyType(np.float32),
                rng_variant.width),
                    -2: ctr_dtype, 0: ctr_dtype, 1:
                    key_dtype}
            return (
                    self.copy(arg_id_to_dtype=constantdict(new_arg_id_to_dtype),
                              name_in_target=name),
                    clbl_inf_ctx)

        elif name == fn + "_f64":
            new_arg_id_to_dtype = {-1: target.vector_dtype(NumpyType(np.float64),
                rng_variant.width),
                    -2: ctr_dtype, 0: ctr_dtype, 1:
                    key_dtype}
            return (
                    self.copy(arg_id_to_dtype=constantdict(new_arg_id_to_dtype),
                              name_in_target=name),
                    clbl_inf_ctx)

        return (self.copy(arg_id_to_dtype=constantdict(arg_id_to_dtype)),
                clbl_inf_ctx)

    def generate_preambles(self, target):
        rng_variant = FUNC_NAMES_TO_RNG[self.name]

        from loopy.target.pyopencl import PyOpenCLTarget
        yield ("90-random123-"+rng_variant.full_name,
                PREAMBLE_TEMPLATE.render(
                    is_pyopencl_target=isinstance(
                        target,
                        PyOpenCLTarget),
                    rng_variant=rng_variant,
                    ))

        return


def get_random123_callables(target):
    return {id_: Random123Callable(id_, target=target) for id_ in FUNC_NAMES_TO_RNG}

# vim: foldmethod=marker
