import numpy as np

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401

k = lp.make_kernel([
    "{ [i] : 0 <= i < m }",
    "{ [j] : 0 <= j < length }"],
    """
    for i
        <> rowstart = rowstarts[i]
        <> rowend = rowstarts[i+1]
        <> length = rowend - rowstart
        y[i] = sum(j, values[rowstart+j] * x[colindices[rowstart + j]])
    end
    """, name="spmv")

k = lp.add_and_infer_dtypes(k, {
    "values,x": np.float64, "rowstarts,colindices": k["spmv"].index_dtype
    })
print(lp.generate_code_v2(k).device_code())
