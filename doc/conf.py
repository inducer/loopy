from importlib import metadata
from urllib.request import urlopen


_conf_url = "https://tiker.net/sphinxconfig-v0.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2016, Andreas Klöckner"
release = metadata.version("loopy")
version = ".".join(release.split(".")[:2])

exclude_patterns = ["_build"]

intersphinx_mapping = {
    "cgen": ("https://documen.tician.de/cgen", None),
    "constantdict": ("https://matthiasdiener.github.io/constantdict/", None),
    "genpy": ("https://documen.tician.de/genpy", None),
    "islpy": ("https://documen.tician.de/islpy", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
}

nitpicky = True
nitpick_ignore_regex = [
    ("py:class", ".*ASTType"),
    ("py:class", ".*EllipsisType"),
    # FIXME: add to pytools docs
    ("py:class", ".*ToTagSetConvertible"),
]

sphinxconfig_missing_reference_aliases = {
    "constantdict": "class:constantdict.constantdict",
    # numpy
    "DTypeLike": "obj:numpy.typing.DTypeLike",
    "np.typing.NDArray": "obj:numpy.typing.NDArray",
    "numpy.complex128": "obj:numpy.complex128",
    "numpy.int16": "obj:numpy.int16",
    # pytools
    "Tag": "class:pytools.tag.Tag",
    "TagT": "obj:pytools.tag.TagT",
    "UniqueNameGenerator": "obj:pytools.UniqueNameGenerator",
    # cgen
    "Generable": "class:cgen.Generable",
    # pymbolic
    "ArithmeticExpression": "data:pymbolic.ArithmeticExpression",
    "Expression": "obj:pymbolic.typing.Expression",
    "ExpressionNode": "class:pymbolic.primitives.ExpressionNode",
    "Variable": "class:pymbolic.primitives.Variable",
    "_Expression": "obj:pymbolic.typing.Expression",
    "p.Call": "obj:pymbolic.primitives.Call",
    "p.CallWithKwargs": "obj:pymbolic.primitives.CallWithKwargs",
    "p.Variable": "obj:pymbolic.primitives.Variable",
    # isl
    "isl.BasicSet": "class:islpy.BasicSet",
    "isl.PwAff": "class:islpy.PwAff",
    "isl.Set": "class:islpy.Set",
    "isl.Space": "class:islpy.Space",
    # loopy
    "InameStr": "obj:loopy.typing.InameStr",
    "InameStrSet": "obj:loopy.typing.InameStrSet",
    "KernelIname": "obj:loopy.kernel.data.Iname",
    "LoopNestTree": "obj:loopy.schedule.tools.LoopNestTree",
    "LoopTree": "obj:loopy.schedule.tools.LoopTree",
    "ShapeType": "obj:loopy.typing.ShapeType",
    "ToLoopyTypeConvertible": "obj:loopy.types.ToLoopyTypeConvertible",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
