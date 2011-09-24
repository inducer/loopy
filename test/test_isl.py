import islpy as isl




def test_aff_to_expr():
    s = isl.Space.create_from_names(isl.Context(), ["a", "b"])
    zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(s))
    one = zero.set_constant(1)
    a = zero.set_coefficient(isl.dim_type.in_, 0, 1)
    b = zero.set_coefficient(isl.dim_type.in_, 1, 1)

    x = (5*a + 3*b) % 17 % 5
    print x
    from loopy.symbolic import aff_to_expr
    print aff_to_expr(x)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
