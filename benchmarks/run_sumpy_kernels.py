import numpy as np

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)

import loopy as lp

def _sumpy_kernel_init(dim, order):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    np.random.seed(17)

    knl = LaplaceKernel(dim)
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion
    target_kernels = [knl]
    m_expn = mpole_expn_class(knl, order=order)
    l_expn = local_expn_class(knl, order=order)

    from sumpy import P2EFromSingleBox, E2PFromSingleBox, P2P, E2EFromCSR
    m2l = E2EFromCSR(ctx, m_expn, l_expn)
    m2l.get_translation_loopy_insns()
    return m2l

def _sumpy_kernel_make(m2l):
    loopy_knl = m2l.get_optimized_kernel()
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


class SumpyBenchmarkSuite:

    params = [
        ("m2l", 3, 6)
    ]

    param_names = ['test_name']

    version = 1

    def setup_cache(self):
        data = {}
        for param in self.params:
            self.setup(data, param)
            data[param] = {}
            m2l = _sumpy_kernel_init(param[1], param[2])
            data[param]["setup"] = m2l
            knl = _sumpy_kernel_make(m2l)
            data[param]["instantiated"] = knl
            preprocessed = lp.preprocess_kernel(knl)
            data[param]["preprocessed"] = preprocessed
            scheduled = lp.get_one_scheduled_kernel(preprocessed)
            data[param]["scheduled"] = scheduled
        return data

    def setup(self, data, param):
        logging.basicConfig(level=logging.INFO)
        np.random.seed(17)

    def time_instantiate(self, data, param):
        create_knl = _sumpy_kernel_make(data[param]["setup"])

    def time_preprocess(self, data, param):
        lp.preprocess_kernel(data[param]["instantiated"])

    def time_schedule(self, data, param):
        lp.get_one_scheduled_kernel(data[param]["preprocessed"])

    def time_generate_code(self, data, param):
        lp.generate_code_v2(data[param]["scheduled"])

    time_instantiate.timeout = 600.0
    time_preprocess.timeout = 600.0
    time_schedule.timeout = 600.0
    time_generate_code.timeout = 600.0

    # No warmup is needed
    time_instantiate.warmup_time = 0
    time_preprocess.warmup_time = 0
    time_schedule.warmup_time = 0
    time_generate_code.warmup_time = 0

    # These are expensive operations. Run only once
    time_schedule.number = 1
    time_generate_code.number = 1

    # Run memory benchmarks as well
    peakmem_instantiate = time_instantiate
    peakmem_preprocess = time_preprocess
    peakmem_schedule = time_schedule
    peakmem_generate_code = time_generate_code

