#! /bin/sh

rsync --progress --verbose --archive --delete _build/html/* buster:doc/loopy
