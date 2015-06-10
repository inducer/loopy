#! /bin/bash

# run this from the loopy root directory

rm -Rf dist build

pyinstaller \
  --workpath=build/pyinstaller \
  build-helpers/loopy.spec
