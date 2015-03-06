#! /bin/bash

set -e
set -x

CNT=$(docker create -t -v $(pwd):/mnt centos:6 /mnt/make-linux-build-docker-inner.sh)
echo "working in container $CNT"

docker start -i $CNT

docker cp $CNT:/tmp/build/loopy/dist/loopy $(pwd) || true

docker rm $CNT

