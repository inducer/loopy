#! /bin/sh

NAME="$1"
shift

python $(which loopy) --lang=floopy "$NAME" - "$@"
