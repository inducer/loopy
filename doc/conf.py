import os
from urllib.request import urlopen


_conf_url = "https://tiker.net/sphinxconfig-v0.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2016, Andreas Klöckner"

# The short X.Y version.
ver_dic = {}
_version_source = "../loopy/version.py"
with open(_version_source) as vpy_file:
    version_py = vpy_file.read()

os.environ["AKPYTHON_EXEC_IMPORT_UNAVAILABLE"] = "1"
exec(compile(version_py, _version_source, "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]
del os.environ["AKPYTHON_EXEC_IMPORT_UNAVAILABLE"]

exclude_patterns = ["_build"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pytools": ("https://documen.tician.de/pytools", None),
        "islpy": ("https://documen.tician.de/islpy", None),
        "pyopencl": ("https://documen.tician.de/pyopencl", None),
        "cgen": ("https://documen.tician.de/cgen", None),
        "genpy": ("https://documen.tician.de/genpy", None),
        "pymbolic": ("https://documen.tician.de/pymbolic", None),
        "constantdict": ("https://matthiasdiener.github.io/constantdict/", None),
        }

nitpicky = True

nitpick_ignore_regex = [
        ["py:class", r"typing_extensions\.(.+)"],
        ["py:class", r"numpy\.u?int[0-9]+"],
        ["py:class", r"numpy\.float[0-9]+"],
        ["py:class", r"numpy\.complex[0-9]+"],

        # Reference not found from "<unknown>"? I'm not even sure where to look.
        ["py:class", r"ExpressionNode"],

        # Type aliases
        ["py:class", r"InameStr"],
        ["py:class", r"ConcreteCallablesTable"],
        ["py:class", r"LoopNestTree"],
        ["py:class", r"LoopTree"],
        ["py:class", r"ToLoopyTypeConvertible"],
        ["py:class", r"ToStackMatchConvertible"],
        ]
