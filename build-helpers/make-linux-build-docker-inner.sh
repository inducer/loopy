#! /bin/bash

set -e
set -x

mkdir /tmp/build
cd /tmp/build

useradd -d /home/user -m -s /bin/bash user

yum install -y centos-release-scl
yum install -y git python27 python27-python-devel python27-numpy tar gcc gcc-c++ mercurial libffi-devel

scl enable python27 /mnt/make-linux-build-docker-inner-part-2.sh

