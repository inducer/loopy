# -*- mode: python -*-

block_cipher = None
single_file = True


a = Analysis(['bin/loopy'],
             pathex=['/home/andreas/src/loopy'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=["hedge", "meshpy", "pyopencl", "pycparser", "PIL"],
             cipher=block_cipher)
pyz = PYZ(a.pure,
             cipher=block_cipher)

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
