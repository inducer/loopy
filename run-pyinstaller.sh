#! /bin/bash

rm -Rf dist/loopy

pyinstaller \
  --workpath=build/pyinstaller \
  loopy.spec
