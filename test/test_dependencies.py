__copyright__ = "Copyright (C) 2023 Addison Alvey-Blanco"

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

import sys
import loopy as lp
from loopy.kernel.dependency import add_lexicographic_happens_after, \
                                    find_data_dependencies
from loopy.transform.dependency import narrow_dependencies

def test_lex_dependencies():
    knl = lp.make_kernel(
            [
                "{[a,b]: 0<=a,b<7}",
                "{[i,j]: 0<=i,j<n and 0<=a,b<5}",
                "{[k,l]: 0<=k,l<n and 0<=a,b<3}"
                ],
            """
            v[a,b,i,j] = 2*v[a,b,i,j]
            v[a,b,k,l] = 2*v[a,b,k,l]
            """)

    knl = add_lexicographic_happens_after(knl)


def test_find_dependencies():
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

    import numpy as np
    k = lp.add_and_infer_dtypes(k, {
        "values,x": np.float64, "rowstarts,colindices": k["spmv"].index_dtype
        })

    k = find_data_dependencies(k)
    pu.db


def test_narrow_dependencies():
    pu.db
    knl = lp.make_kernel(
            "{ [i,j]: 0 <= i < n }",
            """
            a = i
            b = a
            d = b
            e = i - n
            c = e + b
            """)

    knl = add_lexicographic_happens_after(knl)
    knl = find_data_dependencies(knl)
    knl = narrow_dependencies(knl)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
