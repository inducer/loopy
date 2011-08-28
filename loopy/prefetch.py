from __future__ import division

from pytools import Record, memoize_method
from islpy import dim_type




# {{{ register prefetches

class RegisterPrefetch(Record):
    __slots__ = ["subexprs", "new_names"]

def insert_register_prefetches(kernel):
    reg_pf = {}

    total_loop_count = len(kernel.all_inames())
    known_vars = set()

    unused_index_exprs = set()
    from loopy.symbolic import AllSubscriptExpressionCollector
    asec = AllSubscriptExpressionCollector()

    from pymbolic.mapper.dependency import DependencyMapper

    for tgt, expr in kernel.instructions:
        unused_index_exprs |= asec(expr)
    unused_index_exprs = [
            (iexpr, set(v.name for v in DependencyMapper()(iexpr.index)))
            for iexpr in unused_index_exprs]

    schedule = kernel.schedule[:]

    from loopy.schedule import ScheduledLoop

    sched_index = 0
    loop_count = 0
    while sched_index < len(schedule):
        sched_item = schedule[sched_index]
        if isinstance(sched_item, ScheduledLoop):
            known_vars.add(sched_item.iname)
            loop_count += 1
        sched_index += 1

        if loop_count < total_loop_count:
            i = 0
            while i < len(unused_index_exprs):
                iexpr, index_deps = unused_index_exprs[i]
                if (index_deps <= known_vars
                        and (iexpr.aggregate.name, iexpr.index)
                            not in kernel.prefetch):
                    unused_index_exprs.pop(i)
                    new_name = "reg_prefetch_"+iexpr.aggregate.name+str(sched_index)
                    reg_pf[iexpr] = new_name
                    schedule.insert(sched_index,
                            RegisterPrefetch(
                                subexprs=[iexpr], new_names=[new_name]))
                    sched_index += 1
                else:
                    i += 1

    return kernel.copy(schedule=schedule, register_prefetch=reg_pf)

# }}}

# {{{ local-mem prefetch-related

class LocalMemoryPrefetch(Record):
    """
    Attributes:
    :ivar kernel:
    :ivar input_vector: A string indicating the input vector variable name.
    :ivar index_expr: An expression identifying the access which this prefetch
      serves.
    :ivar fetch_dims: A sequence of tuples of inames (i.e. loop dimensions)
        identifying which part of the input vector, given the index_expr, should
        be prefetched. Non-length-1 tuples indicate that these indices should
        share a dimension in the prefetch array.
    :ivar loc_fetch_axes: dictionary from integers 0..len(inames) to lists of
      local index axes which should be used to realize that dimension of the
      prefetch. The last dimension in this list is used as the fastest-changing
      one.
    :ivar name: the variable name used for the prefetch
    :ivar dim_storage_lengths: a sequence of integers indicating the size of
        the storage for each dimension. It may may differ from the size of the
        actual loop dimensions to mitigate bank conflicts.

    The latter two values are only assigned during code generation.
    """

    @memoize_method
    def all_inames(self):
        """Order matters as this will be the order of indices into the
        prefetch array.
        """
        return [
                iname
                for fetch_dim in self.fetch_dims
                for iname in fetch_dim]

    @property
    @memoize_method
    def domain(self):
        return (self.kernel.domain
                .project_out_except(self.all_inames(), [dim_type.set])
                .compute_divs()
                .remove_divs_of_dim_type(dim_type.set))

    @property
    @memoize_method
    def index_map(self):
        from loopy.isl import make_index_map
        imap = make_index_map(self.kernel_domain, self.index_expr)
        assert imap.is_bijective()
        return imap

    @property
    @memoize_method
    def restricted_index_map(self):
        return self.index_map.intersect_domain(self.domain)

    @memoize_method
    def get_dim_bounds_constraints_by_iname(self, iname):
        from loopy.codegen.bounds import get_bounds_constraints
        lower, upper, equality = get_bounds_constraints(
                self.domain, iname, (iname,),
                allow_parameters=False)

        assert not equality

        lower, = lower
        upper, = upper
        return lower, upper

    @property
    @memoize_method
    def dim_bounds_by_iname(self):
        from loopy.codegen.bounds import solve_constraint_for_bound
        result = {}
        for iname in self.all_inames():
            lower, upper = self.get_dim_bounds_constraints_by_iname(iname)

            lower_kind, lower_bound = solve_constraint_for_bound(lower, iname)
            upper_kind, upper_bound = solve_constraint_for_bound(upper, iname)

            try:
                lower_bound = int(lower_bound)
                upper_bound = int(upper_bound)
            except TypeError:
                raise RuntimeError("loop bounds for prefetch must be known statically")

            result[iname] = (lower_bound, upper_bound)

        return result

    @property
    def itemsize(self):
        return self.kernel.arg_dict[self.input_vector].dtype.itemsize

    def dim_lengths(self):
        result = []
        for fetch_dim in self.fetch_dims:
            fd_result = 1
            for iname in fetch_dim:
                start, stop = self.dim_bounds_by_iname[iname]
                fd_result *= stop-start
            result.append(fd_result)
        return result

    @property
    @memoize_method
    def nbytes(self):
        from pytools import product
        return self.itemsize * product(self.dim_lengths())

    @memoize_method
    def free_variables(self):
        from pymbolic.mapper.dependency import DependencyMapper
        return set(var.name
                for var in DependencyMapper()(self.index_expr)
                ) - set(self.all_inames()) - set(self.kernel.scalar_loop_args)

    def hash(self):
        return (hash(type(self)) ^ hash(self.input_vector)
                ^ hash(self.index_expr))

    def __eq__(self, other):
        # in particular, dim_storage_lengths should not factor into equality
        return (type(self) == type(other)
                and self.input_vector == other.input_vector
                and self.index_expr == other.index_expr)

# }}}

# vim: foldmethod=marker
