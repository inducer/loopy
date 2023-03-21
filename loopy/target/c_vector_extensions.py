import numpy as np
from cgen import Declarator
from pytools import memoize_method
from loopy.target import VectorizationFallback
from loopy.target.c import CTarget, CWithGNULibcASTBuilder, ExecutableCTarget
from loopy.types import NumpyType
from loopy.kernel.array import (ArrayBase, FixedStrideArrayDimTag,
                                VectorArrayDimTag)


# {{{ vector types

class vec:  # noqa
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    vec.types = {}
    vec.names_and_dtypes = []
    vec.type_to_scalar_and_count = {}

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [
            ("char", np.int8),
            ("unsigned char", np.uint8),
            ("short", np.int16),
            ("unsigned short", np.uint16),
            ("int", np.int32),
            ("unsigned int", np.uint32),
            ("long", np.int64),
            ("unsigned long", np.uint64),
            ("float", np.float32),
            ("double", np.float64),
            ]:
        for count in counts:
            byte_count = count*np.dtype(base_type).itemsize
            name = "%s __attribute__((vector_size(%d)))" % (base_name,
                    byte_count)

            titles = field_names[:count]

            names = [f"s{i}" for i in range(count)]

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type]*count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            setattr(vec, name, dtype)

            vec.names_and_dtypes.append((name, dtype))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count


_create_vector_types()


def _register_vector_types(dtype_registry):
    for name, dtype in vec.names_and_dtypes:
        dtype_registry.get_or_register_dtype(name, dtype)

# }}}


# {{{ target

class CVectorExtensionsTarget(CTarget):
    """A specialized C-target that represents vectorization through GCC/Clang
    language extensions.
    """
    def __init__(self,
                 vec_fallback: VectorizationFallback = VectorizationFallback.UNROLL,
                 fortran_abi=False):
        super().__init__(fortran_abi=fortran_abi)
        self.vec_fallback = vec_fallback

    def get_host_ast_builder(self):
        return CVectorExtensionsASTBuilder(self)

    def get_device_ast_builder(self):
        return CVectorExtensionsASTBuilder(self)

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c99_stdint_types,
                fill_registry_with_c99_complex_types)
        from loopy.target.c import DTypeRegistryWrapper

        result = DTypeRegistry()
        fill_registry_with_c99_stdint_types(result)
        fill_registry_with_c99_complex_types(result)

        _register_vector_types(result)
        return DTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(
                vec.types[base.numpy_dtype, count],
                target=self)

    @property
    def allows_non_constant_indexing_for_vec_types(self):
        return True

    @property
    def broadcasts_scalar_assignment_to_vec_types(self):
        return False

    @property
    def vectorization_fallback(self):
        return self.vec_fallback


class ExecutableCVectorExtensionsTarget(CVectorExtensionsTarget,
                                        ExecutableCTarget):
    def __init__(self,
                 vec_fallback: VectorizationFallback = VectorizationFallback.UNROLL,
                 compiler=None,
                 fortran_abi=False):
        ExecutableCTarget.__init__(self, compiler=compiler, fortran_abi=fortran_abi)
        self.vec_fallback = vec_fallback

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return ExecutableCTarget.get_kernel_executor_cache_key(self, *args, **kwargs)

    def get_kernel_executor(self, t_unit, *args, **kwargs):
        return ExecutableCTarget.get_kernel_executor(self, t_unit, *args, **kwargs)

    @property
    def is_executable(self) -> bool:
        return True

# }}}


# {{{ AST builder

class CVectorExtensionsASTBuilder(CWithGNULibcASTBuilder):
    def add_vector_access(self, access_expr, index):
        return access_expr[index]

    def get_array_base_declarator(self, ary: ArrayBase) -> Declarator:
        from loopy.target.c import POD
        dtype = ary.dtype
        vec_size = ary.vector_size(self.target)
        if vec_size > 1:
            dtype = self.target.vector_dtype(dtype, vec_size)

        if ary.dim_tags:
            for dim_tag in ary.dim_tags:
                if isinstance(dim_tag, (FixedStrideArrayDimTag,
                                        VectorArrayDimTag)):
                    # we're OK with that
                    pass
                else:
                    raise NotImplementedError(
                        f"{type(self).__name__} does not understand axis tag "
                        f"'{type(dim_tag)}.")

        arg_decl = POD(self, dtype, ary.name)
        return arg_decl

# }}}
