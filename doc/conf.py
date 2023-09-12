import os
from urllib.request import urlopen

_conf_url = "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"  # noqa
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
        "pymbolic": ("https://documen.tician.de/pymbolic", None),
        "pytools": ("https://documen.tician.de/pytools", None),
        "pyrsistent": ("https://pyrsistent.readthedocs.io/en/latest/", None),
        }

# Some modules need to import things just so that sphinx can resolve symbols in
# type annotations. Often, we do not want these imports (e.g. of PyOpenCL) when
# in normal use (because they would introduce unintended side effects or hard
# dependencies). This flag exists so that these imports only occur during doc
# build. Since sphinx appears to resolve type hints lexically (as it should),
# this needs to be cross-module (since, e.g. an inherited arraycontext
# docstring can be read by sphinx when building meshmode, a dependent package),
# this needs a setting of the same name across all packages involved, that's
# why this name is as global-sounding as it is.
import sys
sys._BUILDING_SPHINX_DOCS = True

nitpicky = True

nitpick_ignore_regex = [
        ["py:class", r"typing_extensions\.(.+)"],
        ["py:class", r"numpy\.u?int[0-9]+"],
        ["py:class", r"numpy\.float[0-9]+"],
        ["py:class", r"numpy\.complex[0-9]+"],

        # As of 2022-06-22, it doesn't look like there's sphinx documentation
        # available.
        ["py:class", r"immutables\.(.+)"],
        ]
