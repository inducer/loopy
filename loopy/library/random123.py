"""Library integration with Random123."""


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


from pytools import ImmutableRecord
from mako.template import Template
from loopy.kernel.function_interface import ScalarCallable
import numpy as np


# {{{ rng metadata

class RNGInfo(ImmutableRecord):
    @property
    def full_name(self):
        return "%s%dx%d" % (self.name, self.width, self.bits)


_philox_base_info = RNGInfo(
            name="philox",
            pyopencl_header="pyopencl-random123/philox.cl",
            generic_header="Random123/philox.h",
            key_width=2)

_threefry_base_info = RNGInfo(
            name="threefry",
            pyopencl_header="pyopencl-random123/threefry.cl",
            generic_header="Random123/threefry.h",
            key_width=4)

RNG_VARIANTS = [
        _philox_base_info.copy(width=2, bits=32),
        _philox_base_info.copy(width=2, bits=64),
        _philox_base_info.copy(width=4, bits=32),
        _philox_base_info.copy(width=4, bits=64),

        _threefry_base_info.copy(width=2, bits=32),
        _threefry_base_info.copy(width=2, bits=64),
        _threefry_base_info.copy(width=4, bits=32),
        _threefry_base_info.copy(width=4, bits=64),
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


class Random123Callable(ScalarCallable):
    """
    Records information about for the random123 functions.
    """
    fields = ScalarCallable.fields | {"target"}
    hash_fields = ScalarCallable.hash_fields + ("target",)

    def __init__(self, name, arg_id_to_dtype=None,
                 arg_id_to_descr=None, name_in_target=None, target=None):
        super().__init__(name=name,
                         arg_id_to_dtype=arg_id_to_dtype,
                         arg_id_to_descr=arg_id_to_descr,
                         name_in_target=name_in_target)

        self.target = target

    def with_types(self, arg_id_to_dtype, callables_table):

        if 0 not in arg_id_to_dtype or 1 not in arg_id_to_dtype or (
                arg_id_to_dtype[0] is None or arg_id_to_dtype[1] is None):
            # the types provided aren't mature enough to specialize the
            # callable
            return (self.copy(),
                    callables_table)

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
                    self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
                        name_in_target=fn+"_gen"),
                    callables_table)

        elif name == fn + "_f32":
            new_arg_id_to_dtype = {-1: target.vector_dtype(NumpyType(np.float32),
                rng_variant.width),
                    -2: ctr_dtype, 0: ctr_dtype, 1:
                    key_dtype}
            return self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
                    name_in_target=name), callables_table

        elif name == fn + "_f64":
            new_arg_id_to_dtype = {-1: target.vector_dtype(NumpyType(np.float64),
                rng_variant.width),
                    -2: ctr_dtype, 0: ctr_dtype, 1:
                    key_dtype}
            return self.copy(arg_id_to_dtype=new_arg_id_to_dtype,
                    name_in_target=name), callables_table

        return (self.copy(arg_id_to_dtype=arg_id_to_dtype),
                callables_table)

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
