lp_knl = lp.make_kernel(
	"{[i,j]: 0<=i,j<n}",
	"c[i,j] = a[i]*b[j]")

lp_knl = lp.add_dtypes(lp_knl, {"a": np.float64, "b": np.float64})
lp_knl = lp.split_iname(lp_knl, "i", 16, outer_tag="g.0", inner_tag="l.0")
lp_knl = lp.split_iname(lp_knl, "j", 16, outer_tag="g.1", inner_tag="l.1")

