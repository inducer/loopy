#! /bin/bash

set -e
set -x

mkdir /tmp/build
cd /tmp/build

yum install -y git python-devel tar gcc gcc-c++ mercurial numpy

VENV_VERSION="virtualenv-1.9.1"
rm -Rf "$VENV_VERSION"
curl -k https://pypi.python.org/packages/source/v/virtualenv/$VENV_VERSION.tar.gz | tar xfz -

VIRTUALENV=virtualenv
$VENV_VERSION/virtualenv.py --no-setuptools .env

#curl -k https://bitbucket.org/pypa/setuptools/raw/bootstrap-py24/ez_setup.py | python -
curl -k https://ssl.tiker.net/software/ez_setup.py | python -
if test "$py_version" = "2.5"; then
  # pip 1.3 is the last release with Python 2.5 support
  hash -r
  which easy_install
  easy_install 'pip==1.3.1'
  PIP="pip --insecure"
else
  #curl -k https://raw.github.com/pypa/pip/1.4/contrib/get-pip.py | python -
  curl http://git.tiker.net/pip/blob_plain/77f959a3ce9cc506efbf3a17290d387d0a6624f5:/contrib/get-pip.py | python -

  PIP="pip"
fi

source .env/bin/activate

pip install pyinstaller
git clone --recursive git://github.com/inducer/loopy
cd loopy

grep -v pyopencl requirements.txt > myreq.txt
pip install -r myreq.txt

./build-helpers/run-pyinstaller.sh
