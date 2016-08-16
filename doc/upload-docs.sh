#! /bin/bash

rsync --verbose --archive --delete _build/html/{.*,*} doc-upload:doc/loopy
