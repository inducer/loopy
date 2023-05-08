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
from loopy.kernel.dependency import add_lexicographic_happens_after
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


def test_scalar_dependencies():
    knl = lp.make_kernel(
            "{ [i]: 0 <= i < n }",
            """
            <> a = 10*i
            b = 9*n + i
            c[0] = i
            """)

    knl = narrow_dependencies(knl)

def test_narrow_simple():
    knl = lp.make_kernel(
            "{ [i,j,k]: 0 <= i,j,k < n }",
            """
            a[i,j] = 2*k
            b[j,k] = i*j + k
            c[k,j] = a[i,j]
            """)

    knl = add_lexicographic_happens_after(knl)
    knl = narrow_dependencies(knl)

def test_narrow_deps_spmv():
    knl = lp.make_kernel([
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
    knl = lp.add_and_infer_dtypes(knl, {
        "values,x": np.float64, "rowstarts,colindices": knl["spmv"].index_dtype
        })
    knl = add_lexicographic_happens_after(knl)
    knl = narrow_dependencies(knl)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
