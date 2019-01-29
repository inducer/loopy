import re
from mako.template import Template
import loopy as lp
from loopy.tools import natsorted


def write_to_python(kernel, filename=None):
    """
    Generates a python code for generating *kernel* for sharing kernels.

    :arg kernel: An instance of :class:`loopy.LoopKernel`
    :arg filename: An instance of :class:`str`. If *None*, then prints the
        python file to *stdout*.
    """

    options = []

    printed_insn_ids = set()
    printed_insn_order = []

    def insert_insn_into_order(insn):
        if insn.id in printed_insn_ids:
            return
        printed_insn_ids.add(insn.id)

        for dep_id in natsorted(insn.depends_on):
            insert_insn_into_order(kernel.id_to_insn[dep_id])

        printed_insn_order.append(insn)

    for insn in kernel.instructions:
        insert_insn_into_order(insn)

    for insn in printed_insn_order:
        option = 'id=%s, ' % insn.id
        if insn.depends_on:
            option += ("dep="+":".join(insn.depends_on)+", ")
        if insn.tags:
            option += ("tags="+":".join(insn.tags)+", ")
        if insn.within_inames:
            option += ("inames="+":".join(insn.within_inames)+", ")
        if isinstance(insn, lp.MultiAssignmentBase):
            if insn.atomicity:
                option += "atomic, "
        elif isinstance(insn, lp.BarrierInstruction):
            option += ("mem_kind=%s, " % insn.mem_kind)
        options.append(option[:-2])

    insn_x_options = zip(printed_insn_order, options)

    python_code = r'''<%! import loopy as lp %>import loopy as lp
    import numpy as np
    <%! tv_scope = {0: 'lp.AddressSpace.PRIVATE', 1: 'lp.AddressSpace.LOCAL',
    2: 'lp.AddressSpace.GLOBAL', lp.auto: 'lp.auto' } %>
    knl = lp.make_kernel(
        [
        % for dom in domains:
        "${str(dom)}",
        % endfor
        ],
        """
        % for insn, opts in insn_x_opts:
        % if isinstance(insn, lp.Assignment):
        ${insn.assignee} = ${insn.expression} {${opts}}
        % elif isinstance(insn, lp.BarrierInstruction):
        ... ${insn.synchronization_kind[0]}barrier{${opts}}
        % else:
        **Not implemented for ${type(insn)}**
        % endif
        %endfor
        """, [
            % for arg in args:
            % if isinstance(arg, lp.ValueArg):
            lp.ValueArg(
                name='${arg.name}', dtype=np.${arg.dtype.numpy_dtype.name}),
            % else:
            lp.GlobalArg(
                name='${arg.name}', dtype=np.${arg.dtype.numpy_dtype.name},
                shape=${arg.shape}, for_atomic=${arg.for_atomic}),
            % endif
            % endfor
            % for tv in temp_vars:
            lp.TemporaryVariable(
                name='${tv.name}', dtype=np.${tv.dtype.numpy_dtype.name},
                shape=${tv.shape}, for_atomic=${tv.for_atomic},
                address_space=${tv_scope[tv.address_space]},
                read_only=${tv.read_only},
                % if tv.initializer is not None:
                initializer=${"np."+str((tv.initializer).__repr__())},
                % endif
                ),
            % endfor
            ], lang_version=${lp.VERSION})'''

    python_code = Template(python_code).render(insn_x_opts=insn_x_options,
            domains=kernel.domains, args=kernel.args,
            temp_vars=[k for k in kernel.temporary_variables.values()])

    python_code = re.sub("\\n    ", "\n", python_code)
    if filename:
        with open(filename, 'w') as f:
            f.write(python_code)
    else:
        print(python_code)
