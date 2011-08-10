from __future__ import division

def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()




# TODO: Reuse of previously split dimensions for prefetch
#   (Or general merging)

# TODO: Try, fix reg. prefetch (DG example) / CSEs
# TODO: Custom reductions per red. axis
# TODO: Functions
# TODO: Common subexpressions
# TODO: Parse ops from string
# FIXME: support non-reductive dimensions
# FIXME: write names should be assigned during scheduling

# TODO: Don't emit spurious barriers (no for scheduled before)
# TODO: Make code more readable

# TODO: Divisibility
# TODO: Try different kernels
# TODO:   - Tricky: Convolution, FD
# TODO: Try, fix indirect addressing
# TODO: User controllable switch for slab opt
# TODO: Separate all-bulk from non-bulk kernels. (maybe?) (#ifdef?)

# TODO: implement efficient div_ceil?
# TODO: why are corner cases inefficient?
# TODO: Use gists
# TODO: Imitate codegen bulk slab handling in bulk slab trials




class LoopyAdvisory(UserWarning):
    pass

# {{{ imported user interface

from loopy.kernel import ScalarArg, ArrayArg, ImageArg

from loopy.kernel import LoopKernel
from loopy.schedule import generate_loop_schedules
from loopy.prefetch import insert_register_prefetches
from loopy.compiled import CompiledKernel, drive_timing_run

# }}}

# {{{ high-level modifiers

def split_dimension(knl, *args, **kwargs):
    return knl.split_dimension(*args, **kwargs)

def get_input_access_descriptors(kernel):
    """Return a dictionary mapping input vectors to
    a list of input access descriptor. An input access
    descriptor is a tuple (input_vec, index_expr).
    """
    from loopy.symbolic import VariableIndexExpressionCollector

    from pytools import flatten
    result = {}
    for ivec in kernel.input_vectors():
        result[ivec] = set(
                (ivec, iexpr)
                for iexpr in flatten(
                    VariableIndexExpressionCollector(ivec)(expression)
                    for lvalue, expression in kernel.instructions
                    ))

    return result

def add_prefetch(kernel, input_access_descr, tags_or_inames, loc_fetch_axes={}):
    """
    :arg input_access_descr: see :func:`get_input_access_descriptors`.
        May also be the name of the variable if there is only one
        reference to that variable.
    :arg tags_or_inames: loop dimensions that are used to carry out the prefetch
    """

    if isinstance(input_access_descr, str):
        var_name = input_access_descr
        var_iads = get_input_access_descriptors(kernel)[var_name]

        if len(var_iads) != 1:
            raise ValueError("input access descriptor for variable %s is "
                    "not unique" % var_name)

        input_access_descr, = var_iads

    inames = [kernel.tag_or_iname_to_iname(s) for s in tags_or_inames]
    ivec, iexpr = input_access_descr

    new_prefetch = getattr(kernel, "prefetch", {}).copy()
    if input_access_descr in new_prefetch:
        raise ValueError("a prefetch descriptor for the input access %s[%s] "
                "already exists" % (ivec, iexpr))

    from loopy.prefetch import LocalMemoryPrefetch
    new_prefetch[input_access_descr] = LocalMemoryPrefetch(
            kernel=kernel,
            input_vector=ivec,
            index_expr=iexpr,
            inames=inames,
            loc_fetch_axes={})

    return kernel.copy(prefetch=new_prefetch)

# }}}





# vim: foldmethod=marker
