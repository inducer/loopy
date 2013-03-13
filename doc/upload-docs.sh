#! /bin/bash

cat > _build/html/.htaccess <<EOF
AuthUserFile /home/andreas/htpasswd
AuthGroupFile /dev/null
AuthName "Pre-Release Documentation"
AuthType Basic

require user iliketoast
EOF

rsync --progress --verbose --archive --delete _build/html/{.*,*} doc-upload:doc/loopy
