import loopy as lp
import numpy as np
import pyopencl as cl
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


def _sumpy_kernel_init(param):
    name, dim, order = param.name, param.dim, param.order
    # TODO: add other kernels
    assert name == "m2l"
    from sumpy.expansion.multipole import (
        LaplaceConformingVolumeTaylorMultipoleExpansion,
    )
    from sumpy.expansion.local import LaplaceConformingVolumeTaylorLocalExpansion
    from sumpy.kernel import LaplaceKernel
    from sumpy import E2EFromCSR

    ctx = cl.create_some_context()
    np.random.seed(17)

    knl = LaplaceKernel(dim)
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion
    m_expn = mpole_expn_class(knl, order=order)
    l_expn = local_expn_class(knl, order=order)

    m2l = E2EFromCSR(ctx, m_expn, l_expn, name="loopy_kernel")
    m2l.get_translation_loopy_insns()
    m2l.ctx = None
    m2l.device = None
    return m2l


def _sumpy_kernel_make(expn, param):
    assert param.name == "m2l"
    loopy_knl = expn.get_optimized_kernel()
    loopy_knl = lp.add_and_infer_dtypes(
        loopy_knl,
        dict(
            tgt_ibox=np.int32,
            centers=np.float64,
            tgt_center=np.float64,
            target_boxes=np.int32,
            src_ibox=np.int32,
            src_expansions=np.float64,
            tgt_rscale=np.float64,
            src_rscale=np.float64,
            src_box_starts=np.int32,
            src_box_lists=np.int32,
        ),
    )
    return loopy_knl


@dataclass(frozen=True)
class Param:
    name: str
    dim: int
    order: int


def cached_data(params):
    data = {}
    np.random.seed(17)
    logging.basicConfig(level=logging.INFO)
    for param in params:
        data[param] = {}
        expn = _sumpy_kernel_init(param)
        data[param]["setup"] = expn
        knl = _sumpy_kernel_make(expn, param)
        knl = lp.preprocess_kernel(knl)
        data[param]["instantiated"] = knl
        scheduled = knl.with_kernel(lp.get_one_scheduled_kernel(knl["loopy_kernel"],
                                               knl.callables_table))
        data[param]["scheduled"] = scheduled
    return data


class SumpyBenchmarkSuite:

    params = [
        Param("m2l", dim=3, order=6),
        Param("m2l", dim=3, order=12),
    ]

    param_names = ["test_name"]

    version = 1

    def setup_cache(self):
        return cached_data(self.params)

    def time_instantiate(self, data, param):
        knl = _sumpy_kernel_make(data[param]["setup"], param)
        lp.preprocess_kernel(knl)

    def time_schedule(self, data, param):
        knl = data[param]["instantiated"]
        knl.with_kernel(lp.get_one_scheduled_kernel(knl["loopy_kernel"],
                                                    knl.callables_table))

    def time_generate_code(self, data, param):
        lp.generate_code_v2(data[param]["scheduled"])

    time_instantiate.timeout = 600.0
    time_schedule.timeout = 600.0
    time_generate_code.timeout = 600.0

    # No warmup is needed
    time_instantiate.warmup_time = 0
    time_schedule.warmup_time = 0
    time_generate_code.warmup_time = 0

    # Run memory benchmarks as well
    peakmem_instantiate = time_instantiate
    peakmem_schedule = time_schedule
    peakmem_generate_code = time_generate_code
