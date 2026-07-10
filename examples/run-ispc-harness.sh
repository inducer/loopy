#! /bin/bash

OMP_PLACES=cores OMP_DISPLAY_ENV=true OMP_SCHEDULE=static python "$@"
