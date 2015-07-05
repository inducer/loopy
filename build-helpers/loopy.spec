# -*- mode: python -*-

single_file = True

from os.path import expanduser

a = Analysis(['bin/loopy'],
             pathex=[expanduser('~/src/loopy')],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=["hedge", "meshpy", "pyopencl", "PIL"]
             )
pyz = PYZ(a.pure)

if single_file:
    exe = EXE(pyz,
              a.scripts,
              a.binaries,
              a.zipfiles,
              a.datas,
              name='loopy',
              debug=False,
              strip=None,
              upx=True,
              console=True)
else:
    exe = EXE(pyz,
              a.scripts,
              exclude_binaries=True,
              name='loopy',
              debug=False,
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
