# FIXME Add copyright header


import sys
import loopy as lp


def test_lex_dependencies():
    knl = lp.make_kernel(
            [
                "{[a,b]:0<=a,b<7}",
                "{[i,j]: 0<=i,j<n and 0<=a,b<5}",
                "{[k,l]: 0<=k,l<n and 0<=a,b<3}"
                ],
            """
            v[a,b,i,j] = 2*v[a,b,i,j]
            v[a,b,k,l] = 2*v[a,b,k,l]
            """)

    from loopy.kernel.dependency import add_lexicographic_happens_after

    add_lexicographic_happens_after(knl)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
