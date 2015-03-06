#! /bin/bash

# run this from the loopy root directory

rm -Rf dist/loopy

pyinstaller \
  --workpath=build/pyinstaller \
  build-helpers/loopy.spec
