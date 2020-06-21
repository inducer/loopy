import numpy as np

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)
from .sumpy_kernels import m2l_3d_order_6

import loopy as lp

class SumpyBenchmarkSuite:

    params = [
        "m2l_3d_order_6"
    ]

    param_names = ['test_name']

    version = 1

    def setup_cache(self):
        data = {}
        for param in self.params:
            self.setup(data, param)
            data[param] = {}
            knl = globals()[param]()
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
        create_knl = globals()[param]
        create_knl()

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

    # Run memory benchmarks as well
    peakmem_instantiate = time_instantiate
    peakmem_preprocess = time_preprocess
    peakmem_schedule = time_schedule
    peakmem_generate_code = time_generate_code

