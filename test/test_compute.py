from __future__ import annotations

import namedisl as nisl
import numpy as np

import loopy as lp
from loopy.transform.compute import compute
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

