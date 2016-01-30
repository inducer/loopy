#! /bin/bash

set -e
set -x

mkdir /tmp/build
cd /tmp/build

useradd -d /home/user -m -s /bin/bash user

yum install -y git python-devel tar gcc gcc-c++ mercurial numpy libffi-devel

VENV_VERSION="virtualenv-1.9.1"
rm -Rf "$VENV_VERSION"
curl -k https://pypi.python.org/packages/source/v/virtualenv/$VENV_VERSION.tar.gz | tar xfz -

VIRTUALENV=virtualenv
$VENV_VERSION/virtualenv.py --system-site-packages --no-setuptools .env

source .env/bin/activate

curl -k https://ssl.tiker.net/software/ez_setup.py | python -
curl -k https://gitlab.tiker.net/inducer/pip/raw/7.0.3/contrib/get-pip.py | python -

pip install packaging

PYTHON_VER=$(python -c 'import sys; print(".".join(str(s) for s in sys.version_info[:2]))')
if test "$PYTHON_VER" = "2.6"; then
  pip install pyinstaller==2.1
else
  pip install pyinstaller
fi

git clone --recursive git://github.com/inducer/loopy
cd loopy

grep -v pyopencl requirements.txt > myreq.txt
pip install -r myreq.txt
python setup.py install

chown -R user /tmp/build

su user -p -c "cd /tmp/build && source .env/bin/activate && cd loopy && ./build-helpers/run-pyinstaller.sh"
