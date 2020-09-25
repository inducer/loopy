#! /bin/bash

set -e

curl -L -O -k https://gitlab.tiker.net/inducer/ci-support/raw/master/build-py-project.sh
source build-py-project.sh

function run_examples()
{
  PATTERN=$1
  CMDLINE=$2
  for i in $(find examples -name "$PATTERN" -print ); do
    echo "-----------------------------------------------------------------------"
    echo "RUNNING $i"
    echo "-----------------------------------------------------------------------"
    dn=$(dirname "$i")
    bn=$(basename "$i")
    (cd $dn; echo $CMDLINE "$bn"; $CMDLINE "$bn")
  done
}

function run_py_examples()
{
  run_examples "*.py" ${PY_EXE}
}
function run_ipynb_examples()
{
  run_examples "*.ipynb" "${PY_EXE} -m nbconvert --to html --execute"
}
function run_floopy_examples()
{
  run_examples "*.floopy" "${PY_EXE} -m loopy"
}

run_py_examples
run_ipynb_examples
run_floopy_examples
