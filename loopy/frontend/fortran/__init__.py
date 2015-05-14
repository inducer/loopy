from __future__ import division, with_statement

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

from loopy.diagnostic import LoopyError


def _extract_define_lines(source):
    lines = source.split("\n")

    import re
    comment_re = re.compile(r"^\s*\!(.*)$")

    remaining_lines = []
    define_lines = []

    in_define_code = False
    for l in lines:
        comment_match = comment_re.match(l)

        if comment_match is None:
            if in_define_code:
                raise LoopyError("non-comment source line in define block")

            remaining_lines.append(l)
            continue

        cmt = comment_match.group(1)
        cmt_stripped = cmt.strip()

        if cmt_stripped == "$loopy begin define":
            if in_define_code:
                raise LoopyError("can't enter transform code twice")
            in_define_code = True

        elif cmt_stripped == "$loopy end define":
            if not in_define_code:
                raise LoopyError("can't leave transform code twice")
            in_define_code = False

        elif in_define_code:
            define_lines.append(cmt)

        else:
            remaining_lines.append(l)

    return "\n".join(remaining_lines), "\n".join(define_lines)


def f2loopy(source, free_form=True, strict=True,
        pre_transform_code=None, transform_code_context=None,
        use_c_preprocessor=False, preprocessor_defines=None,
        file_name="<floopy code>"):
    """
    :arg preprocessor_defines: a list of strings as they might occur after a
        C-style ``#define`` directive, for example ``deg2rad(x) (x/180d0 * 3.14d0)``.
    """
    if use_c_preprocessor:
        try:
            import ply.lex as lex
            import ply.cpp as cpp
        except ImportError:
            raise LoopyError("Using the C preprocessor requires PLY to be installed")

        lexer = lex.lex(cpp)

        from ply.cpp import Preprocessor
        p = Preprocessor(lexer)

        if preprocessor_defines:
            for d in preprocessor_defines:
                p.define(d)

        source, define_code = _extract_define_lines(source)
        if define_code is not None:
            from loopy.tools import remove_common_indentation
            define_code = remove_common_indentation(
                    define_code,
                    require_leading_newline=False)
            def_dict = {}
            def_dict["define"] = p.define

            if pre_transform_code is not None:
                def_dict["_MODULE_SOURCE_CODE"] = pre_transform_code
                exec(compile(pre_transform_code,
                    "<loopy pre-transform code>", "exec"), def_dict)

            def_dict["_MODULE_SOURCE_CODE"] = define_code
            exec(compile(define_code, "<loopy defines>", "exec"), def_dict)

        p.parse(source, file_name)

        tokens = []
        while True:
            tok = p.token()

            if not tok:
                break

            if tok.type == "CPP_COMMENT":
                continue

            tokens.append(tok.value)

        source = "".join(tokens)

    from fparser import api
    tree = api.parse(source, isfree=free_form, isstrict=strict,
            analyze=False, ignore_comments=False)

    from loopy.frontend.fortran.translator import F2LoopyTranslator
    f2loopy = F2LoopyTranslator(file_name)
    f2loopy(tree)

    return f2loopy.make_kernels(pre_transform_code=pre_transform_code,
            transform_code_context=transform_code_context)

# vim: foldmethod=marker
