#! /bin/bash

# should be run in this directory (build-helpers)

if test "$1" = "--nodate"; then
  TGT_NAME=loopy-centos6
else
  TGT_NAME=loopy-centos6-$(date +"%Y-%m-%d")
fi

echo "Generating $TGT_NAME..."

set -e
set -x

docker pull centos:6

CNT=$(docker create -t -v $(pwd):/mnt centos:6 /mnt/make-linux-build-docker-inner.sh)
echo "working in container $CNT"

docker start -i $CNT

docker cp $CNT:/tmp/build/loopy/dist/loopy $(pwd) || true

mv loopy $TGT_NAME

docker rm $CNT

