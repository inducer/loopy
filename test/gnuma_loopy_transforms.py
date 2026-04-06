import loopy as lp


def pick_apart_float_cast(value):
    if isinstance(value, (int, float)):
        return value

    import re
    fval_match = re.match(r"^\(\((float|double)\)\s*(.+)\)$", value)
    if fval_match is None:
        return value

    # tp = fval_match.group(1)
    return float(fval_match.group(2))


def fix_euler_parameters(kernel, p_p0, p_Gamma, p_R):  # noqa: N803
    return lp.fix_parameters(
        kernel,
        p_p0=pick_apart_float_cast(p_p0),
        p_Gamma=pick_apart_float_cast(p_Gamma),
        p_R=pick_apart_float_cast(p_R))


def set_q_storage_format(kernel, name):
    kernel = lp.set_array_axis_names(kernel, name, "i,j,k,field,e")

    kernel = lp.split_array_dim(
        kernel, (name, 3, "F"), 4, auto_split_inames=False)
    return lp.tag_array_axes(kernel, name, "N0,N1,N2,vec,N4,N3")


def set_D_storage_format(kernel):  # noqa: N802
    return lp.tag_array_axes(kernel, "D", "f,f")


def set_up_volume_loop(kernel, Nq):  # noqa: N803
    kernel = lp.fix_parameters(kernel, Nq=Nq)
    kernel = lp.prioritize_loops(kernel, "e,k,j,i")
    kernel = lp.tag_inames(kernel, {"e": "g.0", "j": "l.1", "i": "l.0"})
    return lp.assume(kernel, "elements >= 1")
