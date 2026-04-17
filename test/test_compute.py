from __future__ import annotations

import namedisl as nisl
import numpy as np

import loopy as lp
from loopy.transform.compute_stub import _gather_usage_sites, compute
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


def test_compute_stub_simple_substitution_codegen() -> None:
    knl = lp.make_kernel(
        "{ [i] : 0 <= i < n }",
        """
        u_(is) := u[is]
        out[i] = u_(i)
        """,
        [
            lp.GlobalArg("u", shape=(16,), dtype=np.float32),
            lp.GlobalArg("out", shape=(16,), dtype=np.float32, is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )
    knl = lp.fix_parameters(knl, n=16)

    knl = compute(
        knl,
        "u_",
        compute_map=nisl.make_map("{ [is] -> [i_s] : is = i_s }"),
        storage_indices=["i_s"],
        temporal_inames=[],
        temporary_name="u_tmp",
        temporary_dtype=np.float32,
    )

    code = lp.generate_code_v2(knl).device_code()
    assert "float u_tmp[16]" in code
    assert "u_tmp[i_s] = u[i_s]" in code
    assert "out[i] = u_tmp[i]" in code


def test_compute_stub_repeated_substitution_uses_are_unique() -> None:
    knl = lp.make_kernel(
        "{ [i] : 0 <= i < n }",
        """
        u_(is) := u[is]
        out[i] = u_(i) + u_(i + 1) {id=write_out}
        """,
        [
            lp.GlobalArg("u", shape=(16,), dtype=np.float32),
            lp.GlobalArg("out", shape=(16,), dtype=np.float32, is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )
    knl = lp.fix_parameters(knl, n=16)

    sites = _gather_usage_sites(knl["loopy_kernel"], "u_")

    assert [site.key for site in sites] == [("write_out", 0), ("write_out", 1)]
    assert sites[0].args != sites[1].args


def test_compute_stub_ring_buffer_codegen() -> None:
    ntime = 128
    block_size = 32
    knl = lp.make_kernel(
        "{ [t] : 1 <= t < ntime - 1 }",
        """
        u_hist(ts) := u[ts]
        u_next[t + 1] = 2*u_hist(t) - u_hist(t - 1)
        """,
        [
            lp.GlobalArg("u", dtype=np.float64, shape=(ntime,)),
            lp.GlobalArg("u_next", dtype=np.float64, shape=(ntime,), is_output=True),
        ],
        lang_version=LOOPY_USE_LANGUAGE_VERSION_2018_2,
    )
    knl = lp.fix_parameters(knl, ntime=ntime)
    knl = lp.split_iname(
        knl,
        "t",
        block_size,
        inner_iname="ti",
        outer_iname="to",
    )

    knl = compute(
        knl,
        "u_hist",
        compute_map=nisl.make_map("{ [ts] -> [to, ti, tb] : tb = 32*to + ti - ts }"),
        storage_indices=["tb"],
        inames_to_advance="auto",
        temporary_name="u_time_buf",
        temporary_dtype=np.float64,
    )

    code = lp.generate_code_v2(knl).device_code()
    assert "double u_time_buf[2]" in code
    assert "u_time_buf[tb] = u_time_buf[0]" in code
    assert "u_time_buf[tb] = u[-1 * tb + ti + 32 * to]" in code
    assert "u_next[1 + ti + 32 * to]" in code
