#! /bin/bash

set -e
set -x

VENV_VERSION="virtualenv-1.9.1"
rm -Rf "$VENV_VERSION"
curl -k https://pypi.python.org/packages/source/v/virtualenv/$VENV_VERSION.tar.gz | tar xfz -

$VENV_VERSION/virtualenv.py --system-site-packages --no-setuptools .env

source .env/bin/activate

curl -k https://ssl.tiker.net/software/ez_setup.py | python -
curl -k https://gitlab.tiker.net/inducer/pip/raw/7.0.3/contrib/get-pip.py | python -

pip install packaging

PYTHON_VER=$(python -c 'import sys; print(".".join(str(s) for s in sys.version_info[:2]))')
pip install git+https://github.com/pyinstaller/pyinstaller.git@413c37bec126c0bd26084813593f65128966b4b7

git clone --recursive git://github.com/inducer/loopy
cd loopy

grep -v pyopencl requirements.txt > myreq.txt
pip install -r myreq.txt
python setup.py install

chown -R user /tmp/build

su user -p -c "cd /tmp/build && source .env/bin/activate && cd loopy && ./build-helpers/run-pyinstaller.sh"
