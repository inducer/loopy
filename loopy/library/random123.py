"""Library integration with Random123."""

from __future__ import division, absolute_import

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


from pytools import Record
from mako.template import Template
import numpy as np


# {{{ rng metadata

class RNGInfo(Record):
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

FUNC_NAMES_TO_RNG = dict(
        (v.full_name + suffix, v)
        for v in RNG_VARIANTS
        for suffix in [
            "", "_f32", "_f64",
            ])

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


def random123_preamble_generator(preamble_info):
    for f in preamble_info.seen_functions:
        try:
            rng_variant = FUNC_NAMES_TO_RNG[f.name]
        except KeyError:
            continue

        from loopy.target.pyopencl import PyOpenCLTarget
        yield ("90-random123-"+rng_variant.full_name,
                PREAMBLE_TEMPLATE.render(
                    is_pyopencl_target=isinstance(
                        preamble_info.kernel.target,
                        PyOpenCLTarget),
                    rng_variant=rng_variant,
                    ))


def random123_function_mangler(kernel, name, arg_dtypes):
    try:
        rng_variant = FUNC_NAMES_TO_RNG[name]
    except KeyError:
        return None

    from loopy.types import NumpyType
    target = kernel.target
    base_dtype = {32: np.uint32, 64: np.uint64}[rng_variant.bits]
    ctr_dtype = target.vector_dtype(NumpyType(base_dtype), rng_variant.width)
    key_dtype = target.vector_dtype(NumpyType(base_dtype), rng_variant.key_width)

    from loopy.kernel.data import CallMangleInfo
    fn = rng_variant.full_name
    if name == fn:
        return CallMangleInfo(
                target_name=fn+"_gen",
                result_dtypes=(ctr_dtype, ctr_dtype),
                arg_dtypes=(ctr_dtype, key_dtype))

    elif name == fn + "_f32":
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(
                    target.vector_dtype(NumpyType(np.float32), rng_variant.width),
                    ctr_dtype),
                arg_dtypes=(ctr_dtype, key_dtype))

    elif name == fn + "_f64":
        return CallMangleInfo(
                target_name=name,
                result_dtypes=(
                    target.vector_dtype(NumpyType(np.float64), rng_variant.width),
                    ctr_dtype),
                arg_dtypes=(ctr_dtype, key_dtype))

    else:
        return None

# vim: foldmethod=marker
