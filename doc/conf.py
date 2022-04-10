import os
from urllib.request import urlopen

_conf_url = "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
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
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pytools": None,
    "https://documen.tician.de/islpy": None,
    "https://documen.tician.de/pyopencl": None,
    "https://documen.tician.de/cgen": None,
    "https://documen.tician.de/pymbolic": None,
    "https://documen.tician.de/pytools": None,
    "https://pyrsistent.readthedocs.io/en/latest/": None,
    }

nitpick_ignore_regex = [
        ["py:class", r"typing_extensions\.(.+)"],
        ]
