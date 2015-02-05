#! /bin/sh

NAME="$1"
shift

python $(which loopy) --target=cl:0,0 --lang=floopy "$NAME" - "$@"
