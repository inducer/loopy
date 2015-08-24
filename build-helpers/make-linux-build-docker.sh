#! /bin/bash

# should be run in this directory (build-helpers)

set -e
set -x

CNT=$(docker create -t -v $(pwd):/mnt centos:6 /mnt/make-linux-build-docker-inner.sh)
echo "working in container $CNT"

docker start -i $CNT

docker cp $CNT:/tmp/build/loopy/dist/loopy $(pwd) || true

mv loopy loopy-centos6-$(date +"%Y-%m-%d")

docker rm $CNT

