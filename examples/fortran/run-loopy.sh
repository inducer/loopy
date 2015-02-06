#! /bin/sh

python $(which loopy) --lang=loopy "$NAME" - "$@"
