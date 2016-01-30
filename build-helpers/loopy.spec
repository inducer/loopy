# -*- mode: python -*-

from os.path import basename, dirname, join
from glob import glob

single_file = True

# This makes the executable spew debug info.
debug = False

from os.path import expanduser

import packaging # pip install packaging to add

a = Analysis(['../bin/loopy'],
             pathex=[expanduser('~/src/loopy')],
             hiddenimports=[
                "appdirs",
                "packaging.markers",
                "packaging.specifiers",
                "packaging.version",
                ],
             hookspath=None,
             runtime_hooks=None,
             excludes=["hedge", "meshpy", "pyopencl", "PIL"]
             )

import ply.lex
import ply.yacc


a.datas += [
  (join("py-src", "ply", "lex", basename(fn)), fn, "DATA")
  for fn in glob(join(dirname(ply.lex.__file__), "*.py"))
  ] + [
  (join("py-src", "ply", "yacc", basename(fn)), fn, "DATA")
  for fn in glob(join(dirname(ply.yacc.__file__), "*.py"))
  ]

pyz = PYZ(a.pure)

if single_file:
    exe = EXE(pyz,
              a.scripts,
              a.binaries,
              a.zipfiles,
              a.datas,
              name='loopy',
              debug=debug,
              strip=None,
              upx=True,
              console=True)
else:
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='loopy',
              debug=debug,
              strip=None,
              upx=True,
              console=True)
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=None,
                   upx=True,
                   name='loopy')
