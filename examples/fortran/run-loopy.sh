#! /bin/sh

python $(which loopy) --target=cl:0,0 --lang=loopy "$NAME" - "$@"
