import loopy as lp

k = lp.make_kernel([
    "[m] -> { [i] : 0 <= i < m }",
    "[length] -> { [j] : 0 <= j < length }"],
    """
    rowstart = rowstarts[i]
    rowend = rowstarts[1 + i]
    length = rowend + (-1)*rowstart
    rowsum = 0 {id=zerosum}
    rowsum = rowsum + x[-1 + colindices[-1 + rowstart + j]]*values[-1 + rowstart + j] {dep=zerosum}
    y[i] = rowsum
    """)
print(k)
