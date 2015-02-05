# -*- mode: python -*-

block_cipher = None


a = Analysis(['bin/loopy'],
             pathex=['/home/andreas/src/loopy'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=["hedge"],
             cipher=block_cipher)
pyz = PYZ(a.pure,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='loopy',
          debug=False,
          strip=None,
          upx=True,
          console=True )
