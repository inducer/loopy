import numpy as np

# Inspired by a visualization used in the Halide tutorial
# https://www.youtube.com/watch?v=3uiEyEKji0M


def div_ceil(nr, dr):
    return -(-nr // dr)


def product(iterable):
    from operator import mul
    from functools import reduce
    return reduce(mul, iterable, 1)


class ArrayAccessPatternContext:
    def __init__(self, gsize, lsize, subgroup_size=32, decay_constant=0.75):
        self.lsize = lsize
        self.gsize = gsize
        self.subgroup_size = subgroup_size
        self.timestamp = 0
        self.decay_constant = decay_constant

        self.ind_length = len(gsize) + len(lsize)

        self.arrays = []

    def l(self, index):  # noqa: E741,E743
        subscript = [np.newaxis] * self.ind_length
        subscript[len(self.gsize) + index] = slice(None)

        return np.arange(self.lsize[index])[tuple(subscript)]

    def g(self, index):
        subscript = [np.newaxis] * self.ind_length
        subscript[index] = slice(None)

        return np.arange(self.gsize[index])[tuple(subscript)]

    def nsubgroups(self):
        return div_ceil(product(self.lsize), self.subgroup_size)

    def animate(self, f, interval=200):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure()

        plots = []
        for iary, ary in enumerate(self.arrays):
            ax = fig.add_subplot(1, len(self.arrays), 1+iary)
            ax.set_title(ary.name)
            plots.append(ary.plot(ax))

        def data_gen():
            for _ in f():
                self.tick()

                for ary, plot in zip(self.arrays, plots):
                    plot.set_array(ary.get_plot_data())

                fig.canvas.draw()
                yield plots

        # must be kept alive until after plt.show()
        return animation.FuncAnimation(
                fig, lambda x: x, data_gen,
                blit=False, interval=interval, repeat=True)

    def tick(self):
        self.timestamp += 1


class Array:
    def __init__(self, ctx, name, shape, strides, elements_per_row=None):
        # Each array element stores a tuple:
        # (timestamp, subgroup, g0, g1, g2, ) of last acccess

        assert len(shape) == len(strides)

        self.nattributes = 2+len(ctx.gsize)

        if elements_per_row is None:
            if len(shape) > 1:
                minstride = min(strides)
                for sh_i, st_i in zip(shape, strides):
                    if st_i == minstride:
                        elements_per_row = sh_i
                        break
        else:
            elements_per_row = 256

        self.array = np.zeros((product(shape), self.nattributes,), dtype=np.int32)

        self.ctx = ctx
        self.name = name
        self.shape = shape
        self.strides = strides
        self.elements_per_row = elements_per_row

        ctx.arrays.append(self)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        assert len(index) == len(self.shape)

        all_subscript = (np.newaxis,) * self.ctx.ind_length

        def reshape_ind(ind):
            if not isinstance(ind, np.ndarray):
                return ind[all_subscript]

            else:
                assert len(ind.shape) == self.ctx.ind_length

        lin_index = sum(
                ind_i * stride_i
                for ind_i, stride_i in zip(index, self.strides))

        if not isinstance(lin_index, np.ndarray):
            subscript = [np.newaxis] * self.ctx.ind_length
            lin_index = np.array(lin_index)[subscript]

        self.array[lin_index, 0] = self.ctx.timestamp
        for i, _glength in enumerate(self.ctx.gsize):
            if lin_index.shape[i] > 1:
                self.array[lin_index, 2+i] = self.ctx.g(i)

        workitem_index = 0
        for i in range(len(self.ctx.lsize))[::-1]:
            workitem_index = (
                    workitem_index * self.ctx.lsize[i]
                    + self.ctx.l(i))
        subgroup = workitem_index//self.ctx.subgroup_size
        self.array[lin_index, 1] = subgroup

    def __setitem__(self, index, value):
        self.__getitem__(index)

    def get_plot_data(self):
        nelements = self.array.shape[0]
        base_shape = (
                div_ceil(nelements, self.elements_per_row),
                self.elements_per_row,)
        shaped_array = np.zeros(
                base_shape + (self.nattributes,),
                dtype=np.float32)
        shaped_array.reshape(-1, self.nattributes)[:nelements] = self.array

        modulation = np.exp(
                -self.ctx.decay_constant*(self.ctx.timestamp-shaped_array[:, :, 0]))

        subgroup = shaped_array[:, :, 1]
        if self.ctx.nsubgroups() > 1:
            subgroup = subgroup/(self.ctx.nsubgroups()-1)
        else:
            subgroup.fill(1)

        rgb_array = np.zeros(base_shape + (3,))
        if 1:
            if len(self.ctx.gsize) > 1:
                # g.0 -> red
                rgb_array[:, :, 0] = shaped_array[:, :, 2]/(self.ctx.gsize[0]-1)
            if len(self.ctx.gsize) > 1:
                # g.1 -> blue
                rgb_array[:, :, 2] = shaped_array[:, :, 3]/(self.ctx.gsize[1]-1)
        if 1:
            rgb_array[:, :, 1] = subgroup

        return rgb_array*modulation[:, :, np.newaxis]

    def plot(self, ax, **kwargs):
        return ax.imshow(
                self.get_plot_data(), interpolation="nearest",
                **kwargs)


def show_example():
    n = 2**7
    n16 = div_ceil(n, 16)
    ctx = ArrayAccessPatternContext(gsize=(n16, n16), lsize=(16, 16))
    in0 = Array(ctx, "in0", (n, n), (n, 1))

    if 0:
        # knl a
        i_inner = ctx.l(1)
        i_outer = ctx.g(1)
        k_inner = ctx.l(0)

        def f():
            for k_outer in range(n16):
                in0[i_inner + i_outer*16, k_inner + k_outer*16]
                yield
    elif 0:
        # knl b
        j_inner = ctx.l(0)
        j_outer = ctx.g(0)
        k_inner = ctx.l(1)

        def f():
            for k_outer in range(n16):
                in0[k_inner + k_outer*16, j_inner + j_outer*16]
                yield

    ani = ctx.animate(f)
    import matplotlib.pyplot as plt
    if 1:
        plt.show()
    else:
        ani.save("access.mp4")


def show_example_2():
    bsize = 8
    blocks = 3

    ctx = ArrayAccessPatternContext(gsize=(1,), lsize=(1,),
            decay_constant=0.005)
    in0 = Array(ctx, "in0", (blocks*bsize, blocks*bsize), (blocks*bsize, 1))

    def f():
        for i_outer in range(blocks):
            for j_outer in range(blocks):
                for i_inner in range(bsize):
                    for j_inner in range(bsize):
                        in0[i_inner + i_outer*bsize, j_inner + j_outer*bsize]
                        yield

    ani = ctx.animate(f, interval=10)
    import matplotlib.pyplot as plt
    if 1:
        plt.show()
    else:
        ani.save("access.mp4")


if __name__ == "__main__":
    show_example_2()
