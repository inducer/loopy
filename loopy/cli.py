from __future__ import print_function

import sys

import loopy as lp
import numpy as np


def to_python_literal(value):
    try:
        int(value)
    except ValueError:
        pass
    else:
        # It's an integer
        return value

    try:
        float(value)
    except ValueError:
        pass
    else:
        # It's a float
        return repr(float(value))

    if value.endswith("f"):
        try:
            float(value[:-1])
        except ValueError:
            pass
        else:
            # It's a float
            return repr(float(value[:-1]))

    return repr(value)


def defines_to_python_code(defines_str):
    import re
    define_re = re.compile(r"^\#define\s+([a-zA-Z0-9_]+)\s+(.*)$")
    result = []
    for l in defines_str.split("\n"):
        if not l.strip():
            continue

        match = define_re.match(l)
        if match is None:
            raise RuntimeError("#define not understood: '%s'" % l)

        result.append(
                "%s = %s" % (match.group(1), to_python_literal(match.group(2))))

    return "\n".join(result)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Stand-alone loopy frontend")

    parser.add_argument("infile", metavar="INPUT_FILE")
    parser.add_argument("outfile", default="-", metavar="OUTPUT_FILE",
            help="Defaults to stdout ('-').", nargs='?')
    parser.add_argument("--lang", metavar="LANGUAGE", help="loopy|fortran")
    parser.add_argument("--target", choices=(
        "opencl", "ispc", "ispc-occa", "c", "c-fortran", "cuda"),
        default="opencl")
    parser.add_argument("--name")
    parser.add_argument("--transform")
    parser.add_argument("--edit-code", action="store_true")
    parser.add_argument("--occa-defines")
    parser.add_argument("--occa-add-dummy-arg", action="store_true")
    parser.add_argument("--print-ir", action="store_true")
    args = parser.parse_args()

    if args.target == "opencl":
        from loopy.target.opencl import OpenCLTarget
        target = OpenCLTarget()
    elif args.target == "ispc":
        from loopy.target.ispc import ISPCTarget
        target = ISPCTarget()
    elif args.target == "ispc-occa":
        from loopy.target.ispc import ISPCTarget
        target = ISPCTarget(occa_mode=True)
    elif args.target == "c":
        from loopy.target.c import CTarget
        target = CTarget()
    elif args.target == "c-fortran":
        from loopy.target.c import CTarget
        target = CTarget(fortran_abi=True)
    elif args.target == "cuda":
        from loopy.target.cuda import CudaTarget
        target = CudaTarget()
    else:
        raise ValueError("unknown target: %s" % target)

    lp.set_default_target(target)

    lang = None
    if args.infile == "-":
        infile_content = sys.stdin.read()
    else:
        from os.path import splitext
        _, ext = splitext(args.infile)

        lang = {
                ".py": "loopy",
                ".loopy": "loopy",
                ".floopy": "fortran",
                ".f90": "fortran",
                ".fpp": "fortran",
                ".f": "fortran",
                ".f77": "fortran",
                }.get(ext)
        with open(args.infile, "r") as infile_fd:
            infile_content = infile_fd.read()

    if args.lang is not None:
        lang = args.lang

    if lang is None:
        raise RuntimeError("unable to deduce input language "
                "(wrong input file extension? --lang flag?)")

    if lang == "loopy":
        # {{{ path wrangling

        from os.path import dirname, abspath
        from os import getcwd

        infile_dirname = dirname(args.infile)
        if infile_dirname:
            infile_dirname = abspath(infile_dirname)
        else:
            infile_dirname = getcwd()

        sys.path.append(infile_dirname)

        # }}}

        data_dic = {}
        data_dic["lp"] = lp
        data_dic["np"] = np

        if args.occa_defines:
            with open(args.occa_defines, "r") as defines_fd:
                occa_define_code = defines_to_python_code(defines_fd.read())
            exec(compile(occa_define_code, args.occa_defines, "exec"), data_dic)

        with open(args.infile, "r") as infile_fd:
            exec(compile(infile_content, args.infile, "exec"), data_dic)

        if args.transform:
            with open(args.transform, "r") as xform_fd:
                exec(compile(xform_fd.read(),
                    args.transform, "exec"), data_dic)

        try:
            kernel = data_dic["lp_knl"]
        except KeyError:
            raise RuntimeError("loopy-lang requires 'lp_knl' "
                    "to be defined on exit")

        if args.name is not None:
            kernel = kernel.copy(name=args.name)

        kernels = [kernel]

    elif lang in ["fortran", "floopy", "fpp"]:
        pre_transform_code = None
        if args.transform:
            with open(args.transform, "r") as xform_fd:
                pre_transform_code = xform_fd.read()

        if args.occa_defines:
            if pre_transform_code is None:
                pre_transform_code = ""

            with open(args.occa_defines, "r") as defines_fd:
                pre_transform_code = (
                        defines_to_python_code(defines_fd.read())
                        + pre_transform_code)

        kernels = lp.parse_transformed_fortran(
                infile_content, pre_transform_code=pre_transform_code,
                filename=args.infile)

        if args.name is not None:
            kernels = [kernel for kernel in kernels
                    if kernel.name == args.name]

        if not kernels:
            raise RuntimeError("no kernels found (name specified: %s)"
                    % args.name)

    else:
        raise RuntimeError("unknown language: '%s'"
                % args.lang)

    if args.print_ir:
        for kernel in kernels:
            print(kernel, file=sys.stderr)

    if args.occa_add_dummy_arg:
        new_kernels = []
        for kernel in kernels:
            new_args = [
                    lp.GlobalArg("occa_info", np.int32, shape=None)
                    ] + kernel.args
            new_kernels.append(kernel.copy(args=new_args))

        kernels = new_kernels
        del new_kernels

    codes = []
    from loopy.codegen import generate_code
    for kernel in kernels:
        kernel = lp.preprocess_kernel(kernel)
        code, impl_arg_info = generate_code(kernel)
        codes.append(code)

    if args.outfile is not None:
        outfile = args.outfile
    else:
        outfile = "-"

    code = "\n\n".join(codes)

    # {{{ edit code if requested

    import os
    edit_kernel_env = os.environ.get("LOOPY_EDIT_KERNEL")
    need_edit = args.edit_code
    if not need_edit and edit_kernel_env is not None:
        # Do not replace with "any()"--Py2.6/2.7 bug doesn't like
        # comprehensions in functions with exec().

        for k in kernels:
            if edit_kernel_env.lower() in k.name.lower():
                need_edit = True

    if need_edit:
        from pytools import invoke_editor
        code = invoke_editor(code, filename="edit.cl")

    # }}}

    if outfile == "-":
        sys.stdout.write(code)
    else:
        with open(outfile, "w") as outfile_fd:
            outfile_fd.write(code)
