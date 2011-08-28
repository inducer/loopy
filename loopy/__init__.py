from __future__ import division

def register_mpz_with_pymbolic():
    from pymbolic.primitives import register_constant_class
    import gmpy
    mpz_type = type(gmpy.mpz(1))
    register_constant_class(mpz_type)

register_mpz_with_pymbolic()



# Immediately:
# ------------
# TODO: Imitate codegen bulk slab handling in bulk slab trials

# For writeup:
# ------------
# TODO: Try, fix reg. prefetch (DG example) / CSEs
#   ILP and reg. prefetch interact!
# TODO: Custom reductions per red. axis
# TODO: Functions
# TODO: Common subexpressions
# TODO: Array common subexpressions
# FIXME: support non-reductive dimensions (what did I mean here?)
# FIXME: write names should be assigned during scheduling
# FIXME: screwy lower bounds in ILP

# TODO: Divisibility
# TODO: Try, fix indirect addressing

# TODO: Implement GT200 matmul, Fermi matmul, DG
# TODO: DMA engine threads?
# TODO: Deal with equalities that crop up.
# TODO: Better user feedback.

# Later:
# ------
# TODO: Try different kernels
# TODO:   - Tricky: Convolution, Stencil
# TODO: Separate all-bulk from non-bulk kernels. (maybe?) (#ifdef?)
# TODO: implement efficient ceil_div? (as opposed to floor_div)
# TODO: why are corner cases inefficient?
# TODO: Use gists (why do disjoint sets arise?)




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

def add_prefetch(kernel, input_access_descr, fetch_dims, loc_fetch_axes={}):
    """
    :arg input_access_descr: see :func:`get_input_access_descriptors`.
        May also be the name of the variable if there is only one
        reference to that variable.
    :arg fetch_dims: loop dimensions indexing the input variable on which
        the prefetch is to be carried out.
    """

    if isinstance(input_access_descr, str):
        var_name = input_access_descr
        var_iads = get_input_access_descriptors(kernel)[var_name]

        if len(var_iads) != 1:
            raise ValueError("input access descriptor for variable %s is "
                    "not unique" % var_name)

        input_access_descr, = var_iads

    def parse_fetch_dim(iname):
        if isinstance(iname, str):
            iname = (iname,)

        return tuple(kernel.tag_or_iname_to_iname(s) for s in iname)

    fetch_dims = [parse_fetch_dim(fd) for fd in fetch_dims]
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
            fetch_dims=fetch_dims,
            loc_fetch_axes=loc_fetch_axes)

    return kernel.copy(prefetch=new_prefetch)

# }}}





# vim: foldmethod=marker
