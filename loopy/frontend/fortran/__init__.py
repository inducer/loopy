__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

import logging
logger = logging.getLogger(__name__)

from loopy.diagnostic import LoopyError
from pytools import ProcessLogger


def c_preprocess(source, defines=None, filename=None, include_paths=None):
    """
    :arg source: a string, possibly containing C preprocessor constructs
    :arg defines: a list of strings as they might occur after a
        C-style ``#define`` directive, for example ``deg2rad(x) (x/180d0 * 3.14d0)``.
    :return: a string
    """
    try:
        import ply.lex as lex
        import ply.cpp as cpp
    except ImportError:
        raise LoopyError("Using the C preprocessor requires PLY to be installed")

    input_dirname = None
    if filename is None:
        filename = "<floopy source>"
    else:
        from os.path import dirname
        input_dirname = dirname(filename)

    lexer = lex.lex(cpp)

    from ply.cpp import Preprocessor
    p = Preprocessor(lexer)
    if input_dirname is not None:
        p.add_path(input_dirname)
    if include_paths:
        for inc_path in include_paths:
            p.add_path(inc_path)

    if defines:
        for d in defines:
            p.define(d)

    p.parse(source, filename)

    tokens = []
    while True:
        tok = p.token()

        if not tok:
            break

        if tok.type == "CPP_COMMENT":
            continue

        tokens.append(tok.value)

    return "".join(tokens)


def _extract_loopy_lines(source):
    lines = source.split("\n")

    import re
    comment_re = re.compile(r"^\s*\!(.*)$")

    remaining_lines = []
    loopy_lines = []

    in_loopy_code = False
    for line in lines:
        comment_match = comment_re.match(line)

        if comment_match is None:
            if in_loopy_code:
                raise LoopyError("non-comment source line in loopy block")

            remaining_lines.append(line)

            # Preserves line numbers in loopy code, for debuggability
            loopy_lines.append("# "+line)
            continue

        cmt = comment_match.group(1)
        cmt_stripped = cmt.strip()

        if cmt_stripped == "$loopy begin":
            if in_loopy_code:
                raise LoopyError("can't enter loopy block twice")
            in_loopy_code = True

            # Preserves line numbers in loopy code, for debuggability
            loopy_lines.append("# "+line)

        elif cmt_stripped == "$loopy end":
            if not in_loopy_code:
                raise LoopyError("can't leave loopy block twice")
            in_loopy_code = False

            # Preserves line numbers in loopy code, for debuggability
            loopy_lines.append("# "+line)

        elif in_loopy_code:
            loopy_lines.append(cmt)

        else:
            remaining_lines.append(line)

            # Preserves line numbers in loopy code, for debuggability
            loopy_lines.append("# "+line)

    return "\n".join(remaining_lines), "\n".join(loopy_lines)


def parse_transformed_fortran(source, free_form=True, strict=True,
        pre_transform_code=None, transform_code_context=None,
        filename="<floopy code>"):
    """
    :arg source: a string of Fortran source code which must include
        a snippet of transform code as described below.
    :arg pre_transform_code: code that is run in the same context
        as the transform

    *source* may contain snippets of loopy transform code between markers::

        !$loopy begin
        ! ...
        !$loopy end

    Within the transform code, the following symbols are predefined:

    * ``lp``: a reference to the :mod:`loopy` package
    * ``np``: a reference to the :mod:`numpy` package
    * ``SOURCE``: the source code surrounding the transform block.
      This may be processed using :func:`c_preprocess` and
      :func:`parse_fortran`.
    * ``FILENAME``: the file name of the code being processed

    The transform code must define ``RESULT``, conventionally a list of kernels
    or a :class:`loopy.TranslationUnit`, which is returned from this function
    unmodified.

    An example of *source* may look as follows::

        subroutine fill(out, a, n)
          implicit none

          real*8 a, out(n)
          integer n, i

          do i = 1, n
            out(i) = a
          end do
        end

        !$loopy begin
        !
        ! fill, = lp.parse_fortran(SOURCE, FILENAME)
        ! fill = lp.split_iname(fill, "i", split_amount,
        !     outer_tag="g.0", inner_tag="l.0")
        ! RESULT = [fill]
        !
        !$loopy end
    """

    source, transform_code = _extract_loopy_lines(source)
    if not transform_code:
        raise LoopyError("no transform code found")

    from loopy.tools import remove_common_indentation
    transform_code = remove_common_indentation(
            transform_code,
            require_leading_newline=False,
            ignore_lines_starting_with="#")

    if transform_code_context is None:
        proc_dict = {}
    else:
        proc_dict = transform_code_context.copy()

    import loopy as lp
    import numpy as np

    proc_dict["lp"] = lp
    proc_dict["np"] = np

    proc_dict["SOURCE"] = source
    proc_dict["FILENAME"] = filename

    from os.path import dirname, abspath
    from os import getcwd

    infile_dirname = dirname(filename)
    if infile_dirname:
        infile_dirname = abspath(infile_dirname)
    else:
        infile_dirname = getcwd()

    import sys
    prev_sys_path = sys.path
    try:
        if infile_dirname:
            sys.path = prev_sys_path + [infile_dirname]

        if pre_transform_code is not None:
            proc_dict["_MODULE_SOURCE_CODE"] = pre_transform_code
            exec(compile(pre_transform_code,
                "<loopy pre-transform code>", "exec"), proc_dict)

        proc_dict["_MODULE_SOURCE_CODE"] = transform_code
        exec(compile(transform_code, filename, "exec"), proc_dict)

    finally:
        sys.path = prev_sys_path

    if "RESULT" not in proc_dict:
        raise LoopyError("transform code did not set RESULT")

    return proc_dict["RESULT"]


def _add_assignees_to_calls(knl, all_kernels):
    """
    Returns a copy of *knl* coming from the fortran parser adjusted to the
    loopy specification that written variables of a call must appear in the
    assignee.

    :param knl: An instance of :class:`loopy.LoopKernel`, which have incorrect
        calls to the kernels in *all_kernels* by stuffing both the input and
        output arguments into parameters.

    :param all_kernels: An instance of :class:`list` of loopy kernels which
        may be called by *kernel*.
    """
    new_insns = []
    subroutine_dict = {kernel.name: kernel for kernel in all_kernels}
    from loopy.kernel.instruction import (Assignment, CallInstruction,
            CInstruction, _DataObliviousInstruction,
            modify_assignee_for_array_call)
    from pymbolic.primitives import Call, Variable

    for insn in knl.instructions:
        if isinstance(insn, CallInstruction):
            if isinstance(insn.expression, Call) and (
                    insn.expression.function.name in subroutine_dict):
                assignees = []
                new_params = []
                subroutine = subroutine_dict[insn.expression.function.name]
                for par, arg in zip(insn.expression.parameters, subroutine.args):
                    if arg.name in subroutine.get_written_variables():
                        par = modify_assignee_for_array_call(par)
                        assignees.append(par)
                    if arg.name in subroutine.get_read_variables():
                        new_params.append(par)
                    if arg.name not in (subroutine.get_written_variables() |
                            subroutine.get_read_variables()):
                        new_params.append(par)

                new_insns.append(
                        insn.copy(
                            assignees=tuple(assignees),
                            expression=Variable(
                                insn.expression.function.name)(*new_params)))
            else:
                new_insns.append(insn)
            pass
        elif isinstance(insn, (Assignment, CInstruction,
                _DataObliviousInstruction)):
            new_insns.append(insn)
        else:
            raise NotImplementedError(type(insn).__name__)

    return knl.copy(instructions=new_insns)


def parse_fortran(source, filename="<floopy code>", free_form=None, strict=None,
        seq_dependencies=None, auto_dependencies=None, target=None):
    """
    :returns: a :class:`loopy.TranslationUnit`
    """

    parse_plog = ProcessLogger(logger, "parsing fortran file '%s'" % filename)

    if seq_dependencies is not None and auto_dependencies is not None:
        raise TypeError(
                "may not specify both seq_dependencies and auto_dependencies")
    if auto_dependencies is not None:
        from warnings import warn
        warn("auto_dependencies is deprecated, use seq_dependencies instead",
                DeprecationWarning, stacklevel=2)
        seq_dependencies = auto_dependencies

    if seq_dependencies is None:
        seq_dependencies = True
    if free_form is None:
        free_form = True
    if strict is None:
        strict = True

    import logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("fparser").addHandler(console)

    from fparser import api
    tree = api.parse(source, isfree=free_form, isstrict=strict,
            analyze=False, ignore_comments=False)

    if tree is None:
        raise LoopyError("Fortran parser was unhappy with source code "
                "and returned invalid data (Sorry!)")

    from loopy.frontend.fortran.translator import F2LoopyTranslator
    f2loopy = F2LoopyTranslator(filename, target=target)
    f2loopy(tree)

    kernels = f2loopy.make_kernels(seq_dependencies=seq_dependencies)

    from loopy.transform.callable import merge
    prog = merge(kernels)
    all_kernels = [clbl.subkernel
                   for clbl in prog.callables_table.values()]

    for knl in all_kernels:
        prog.with_kernel(_add_assignees_to_calls(knl, all_kernels))

    if len(all_kernels) == 1:
        # guesssing in the case of only one function
        prog = prog.with_entrypoints(all_kernels[0].name)

    from loopy.frontend.fortran.translator import specialize_fortran_division
    prog = specialize_fortran_division(prog)

    parse_plog.done()

    return prog


# vim: foldmethod=marker
