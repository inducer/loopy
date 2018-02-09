#! /bin/bash

set -e

curl -L -O -k https://gitlab.tiker.net/inducer/ci-support/raw/master/build-py-project.sh
source build-py-project.sh

cd examples
for i in $(find . -name '*.py' -print ); do
  echo "-----------------------------------------------------------------------"
  echo "RUNNING $i"
  echo "-----------------------------------------------------------------------"
  dn=$(dirname "$i")
  bn=$(basename "$i")
  (cd $dn; ${PY_EXE} "$bn")
done
